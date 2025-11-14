#!/bin/bash

# Trino LCM Docker実行スクリプト

echo "🐳 Trino LCM Docker環境を起動中..."

# Docker Composeでビルドと実行
docker-compose up --build

echo "✅ 実行完了"



