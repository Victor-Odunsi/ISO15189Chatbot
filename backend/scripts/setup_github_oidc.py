"""
One-time (idempotent) setup: creates the GitHub Actions OIDC identity
provider and the IAM role .github/workflows/deploy-backend.yml assumes to
run `sam deploy` -- no long-lived AWS access keys stored in GitHub.

Reads AWS credentials from the environment (source your .env first).
Safe to re-run: creation calls are guarded against "already exists".
"""
import json
import os

import boto3
from botocore.exceptions import ClientError

GITHUB_REPO = "Victor-Odunsi/ISO15189Chatbot"
GITHUB_BRANCH = "main"
ROLE_NAME = "iso15189-github-deploy"
PROVIDER_URL = "token.actions.githubusercontent.com"

MANAGED_POLICIES = [
    "arn:aws:iam::aws:policy/AWSCloudFormationFullAccess",
    "arn:aws:iam::aws:policy/AWSLambda_FullAccess",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    "arn:aws:iam::aws:policy/AmazonSQSFullAccess",
    "arn:aws:iam::aws:policy/IAMFullAccess",
]


def main():
    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    iam = session.client("iam")
    sts = session.client("sts")

    # 1. OIDC provider -- thumbprint not required, AWS verifies well-known
    #    providers like this one via trusted CAs.
    try:
        resp = iam.create_open_id_connect_provider(
            Url=f"https://{PROVIDER_URL}",
            ClientIDList=["sts.amazonaws.com"],
        )
        provider_arn = resp["OpenIDConnectProviderArn"]
        print("Created OIDC provider:", provider_arn)
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            account_id = sts.get_caller_identity()["Account"]
            provider_arn = f"arn:aws:iam::{account_id}:oidc-provider/{PROVIDER_URL}"
            print("OIDC provider already existed:", provider_arn)
        else:
            raise

    # 2. Trust policy scoped to this exact repo, this branch only
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Federated": provider_arn},
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {f"{PROVIDER_URL}:aud": "sts.amazonaws.com"},
                "StringLike": {f"{PROVIDER_URL}:sub": f"repo:{GITHUB_REPO}:ref:refs/heads/{GITHUB_BRANCH}"},
            },
        }],
    }

    try:
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description=f"Assumed by GitHub Actions ({GITHUB_REPO}, {GITHUB_BRANCH} branch only) to run sam deploy",
        )
        print("Created role:", role["Role"]["Arn"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            iam.update_assume_role_policy(RoleName=ROLE_NAME, PolicyDocument=json.dumps(trust_policy))
            role = iam.get_role(RoleName=ROLE_NAME)
            print("Role already existed, trust policy updated:", role["Role"]["Arn"])
        else:
            raise

    # 3. Same managed-policy scope as the human IAM user
    for policy_arn in MANAGED_POLICIES:
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy_arn)
    print(f"Attached {len(MANAGED_POLICIES)} managed policies")
    print()
    print("ROLE ARN (put this in the GitHub Actions AWS_DEPLOY_ROLE_ARN secret):")
    print(role["Role"]["Arn"])


if __name__ == "__main__":
    main()
