#!/bin/bash

# 新しいプライベートリポジトリを作成するスクリプト

cd "$(dirname "$0")"

echo "新しいプライベートリポジトリを作成しています..."

# GitHub CLIで新しいプライベートリポジトリを作成
# リモート名は 'new-origin' として追加
gh repo create lcm-eval-copy \
  --private \
  --source=. \
  --remote=new-origin \
  --description "LCM evaluation copy - private repository"

if [ $? -eq 0 ]; then
  echo ""
  echo "✅ プライベートリポジトリが正常に作成されました！"
  echo "リモート名: new-origin"
  echo ""
  echo "次のステップ:"
  echo "1. 変更をコミット: git add . && git commit -m 'Initial commit for private repo'"
  echo "2. プッシュ: git push -u new-origin main"
else
  echo ""
  echo "❌ リポジトリ作成に失敗しました。"
  echo "GitHub CLIでログインしてください: gh auth login"
fi

