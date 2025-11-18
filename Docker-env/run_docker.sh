#!/bin/bash

# Trino LCM Docker実行スクリプト

echo "🐳 Trino LCM Docker環境を起動中..."

# Docker Composeでビルドと実行 (v1 固定運用)
if ! command -v docker-compose >/dev/null 2>&1; then
  echo "❌ docker-compose コマンドが見つかりません。"
  echo "   例: sudo apt-get install docker-compose"
  exit 1
fi

docker-compose up --build

echo "✅ 実行完了"



