# Open WebUI Extensions

このリポジトリは、Open WebUI用の個人的なツール・拡張機能を含んでいます。

## 注意事項

- 他の人が使用することは現状想定していません

## 含まれているもの

- **Memory**: Open WebUIのメモリ機能を拡張するツール
- **Universal File Generator**: 様々な形式のファイル生成ツール（Pandoc版含む）
- **Graphiti Memory**: [Graphiti](https://github.com/getzep/graphiti)を利用したナレッジグラフベースのメモリ拡張機能
  - 別リポジトリ [open-webui-graphiti-memory](https://github.com/Skyzi000/open-webui-graphiti-memory) として管理
  - このリポジトリでは `graphiti/` サブモジュールから参照
  - ファイル:
    - Filter: [graphiti/functions/filter/graphiti_memory.py](graphiti/functions/filter/graphiti_memory.py)
    - Tool: [graphiti/tools/graphiti_memory_manage.py](graphiti/tools/graphiti_memory_manage.py)
    - Action: [graphiti/functions/action/add_graphiti_memory_action.py](graphiti/functions/action/add_graphiti_memory_action.py)

## セットアップ

### サブモジュールの初期化

このリポジトリをクローンした後、サブモジュールを初期化する必要があります:

```bash
git submodule init
git submodule update
```

または、クローン時に一度に実行:

```bash
git clone --recurse-submodules https://github.com/Skyzi000/open-webui-extensions.git
```

## ライセンス

※Skyzi000が作成したコード部分にのみ適用されます。

MIT License
