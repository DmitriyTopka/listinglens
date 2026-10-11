#!/usr/bin/env bash
# Deploy ListingLens to AWS Lambda (arm64 = Graviton) with a public Function URL.
# Run in AWS CloudShell (or any shell with AWS CLI v2 + credentials) from the folder that holds
# function.zip, layer-numpy.zip, layer-opencv.zip (built by deploy/build_lambda.sh).
#
# Everything here stays inside the AWS always-free allowances:
#   Lambda 1M requests + 400k GB-s / month, CloudWatch 10 alarms + 5 GB logs, SNS 1M publishes.
# A kill switch caps abuse: if the function is invoked more than $HOURLY_CAP times in an hour,
# a CloudWatch alarm -> SNS -> tiny "killswitch" Lambda sets its concurrency to 0 (service off).
set -euo pipefail

REGION=${AWS_REGION:-us-east-1}
FN=listinglens
ROLE=listinglens-lambda-role
KROLE=listinglens-killswitch-role
HOURLY_CAP=${HOURLY_CAP:-300}
ALERT_EMAIL=${ALERT_EMAIL:-}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export AWS_DEFAULT_REGION=$REGION
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

ensure_role() {  # name
  if ! aws iam get-role --role-name "$1" >/dev/null 2>&1; then
    aws iam create-role --role-name "$1" --assume-role-policy-document "$TRUST" >/dev/null
    aws iam attach-role-policy --role-name "$1" \
      --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    echo "created role $1"; sleep 10   # IAM propagation
  fi
}

echo "== roles"
ensure_role "$ROLE"
ensure_role "$KROLE"
aws iam put-role-policy --role-name "$KROLE" --policy-name stop-listinglens --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"lambda:PutFunctionConcurrency\",\"Resource\":\"arn:aws:lambda:$REGION:$ACCOUNT:function:$FN\"}]}"

echo "== layers (arm64, python3.12)"
L_NP=$(aws lambda publish-layer-version --layer-name listinglens-numpy --zip-file fileb://layer-numpy.zip \
  --compatible-runtimes python3.12 --compatible-architectures arm64 --query LayerVersionArn --output text)
L_CV=$(aws lambda publish-layer-version --layer-name listinglens-opencv5 --zip-file fileb://layer-opencv.zip \
  --compatible-runtimes python3.12 --compatible-architectures arm64 --query LayerVersionArn --output text)
echo "$L_NP"; echo "$L_CV"

echo "== function"
if aws lambda get-function --function-name "$FN" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FN" --zip-file fileb://function.zip >/dev/null
  aws lambda wait function-updated --function-name "$FN"
  aws lambda update-function-configuration --function-name "$FN" --layers "$L_NP" "$L_CV" \
    --memory-size 2048 --timeout 30 --environment "Variables={LL_PLANNER=rules}" >/dev/null
else
  aws lambda create-function --function-name "$FN" --runtime python3.12 --architectures arm64 \
    --handler lambda_function.handler --role "arn:aws:iam::$ACCOUNT:role/$ROLE" \
    --zip-file fileb://function.zip --layers "$L_NP" "$L_CV" --memory-size 2048 --timeout 30 \
    --environment "Variables={LL_PLANNER=rules}" >/dev/null
fi
aws lambda wait function-updated --function-name "$FN" 2>/dev/null || aws lambda wait function-active --function-name "$FN"

echo "== logs: keep 14 days"
aws logs create-log-group --log-group-name "/aws/lambda/$FN" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "/aws/lambda/$FN" --retention-in-days 14

echo "== public HTTPS Function URL"
if ! aws lambda get-function-url-config --function-name "$FN" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$FN" --auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FN" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FN" --statement-id public-url-invoke \
    --action lambda:InvokeFunction --principal '*' --invoked-via-function-url >/dev/null 2>&1 || true
fi
URL=$(aws lambda get-function-url-config --function-name "$FN" --query FunctionUrl --output text)

echo "== kill switch (> $HOURLY_CAP invocations/hour -> concurrency 0)"
cat > /tmp/killswitch.py <<PY
import boto3
def handler(event, context):
    boto3.client("lambda").put_function_concurrency(FunctionName="$FN", ReservedConcurrentExecutions=0)
    print("listinglens disabled by kill switch")
PY
(cd /tmp && rm -f killswitch.zip && zip -q killswitch.zip killswitch.py)
if ! aws lambda get-function --function-name listinglens-killswitch >/dev/null 2>&1; then
  aws lambda create-function --function-name listinglens-killswitch --runtime python3.12 --architectures arm64 \
    --handler killswitch.handler --role "arn:aws:iam::$ACCOUNT:role/$KROLE" --zip-file fileb:///tmp/killswitch.zip \
    --timeout 10 --memory-size 128 >/dev/null
fi
TOPIC=$(aws sns create-topic --name listinglens-killswitch --query TopicArn --output text)
KARN="arn:aws:lambda:$REGION:$ACCOUNT:function:listinglens-killswitch"
aws lambda add-permission --function-name listinglens-killswitch --statement-id sns-invoke \
  --action lambda:InvokeFunction --principal sns.amazonaws.com --source-arn "$TOPIC" >/dev/null 2>&1 || true
aws sns subscribe --topic-arn "$TOPIC" --protocol lambda --notification-endpoint "$KARN" >/dev/null
if [ -n "$ALERT_EMAIL" ] && ! aws sns list-subscriptions-by-topic --topic-arn "$TOPIC" \
    --query "Subscriptions[?Endpoint=='$ALERT_EMAIL']" --output text | grep -q .; then
  aws sns subscribe --topic-arn "$TOPIC" --protocol email --notification-endpoint "$ALERT_EMAIL" >/dev/null
fi
aws cloudwatch put-metric-alarm --alarm-name listinglens-hourly-cap \
  --namespace AWS/Lambda --metric-name Invocations --dimensions Name=FunctionName,Value=$FN \
  --statistic Sum --period 3600 --evaluation-periods 1 --threshold "$HOURLY_CAP" \
  --comparison-operator GreaterThanThreshold --treat-missing-data notBreaching --alarm-actions "$TOPIC"

echo "== smoke test"
curl -s "${URL}health"; echo
echo "DONE  $URL"
