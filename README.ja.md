# Open WebUI Extensions

このリポジトリは、Open WebUI用のツールとフィルターを含んでいます。

## 🌟 ハイライト

**[Sub Agent Tool](tools/sub_agent.py)** ([openwebui.com](https://openwebui.com/posts/sub_agent_7bfeb0b7)) - openwebui.com で **Upvote数1位**、ダウンロード数 **7,500+** を達成！ Open WebUI 公式の [Community Newsletter, January 28th 2026](https://openwebui.com/blog/newsletter-january-28-2026) で "This Week's Most Useful" ツールの1つとして紹介されました。

ツール呼び出しが多いタスクをサブエージェントに委譲し、メインの会話コンテキストをクリーンに保つツールです。Open WebUI v0.7 以降のビルトインツール（Web検索、メモリ、ナレッジベース等）を最大限に活用できます。

> [!TIP]
> v0.3以降、`run_parallel_sub_agents` によるサブエージェントの並列実行をネイティブサポートしています。サブエージェントの並列実行のために「Parallel Tools」を併用する必要はなくなりました。
> 並列実行で問題が発生する場合（検索APIのレートリミット等）は、Valvesの `MAX_PARALLEL_AGENTS` を下げるか、`run_parallel_sub_agents` メソッドをコメントアウトして無効化してください。

<!-- markdownlint-disable-next-line MD028 -->

> [!TIP]
> v0.4以降、Open WebUI v0.8 で導入されたスキルが自動的にサブエージェントへ伝播されます（実験的機能）。

<!-- markdownlint-disable-next-line MD028 -->

> [!TIP]
> v0.4.5以降、Open Terminal のツール（Open WebUI v0.8.6+）が自動的にサブエージェントに転送されます。

<!-- markdownlint-disable-next-line MD028 -->

> [!TIP]
> v0.5以降、Open WebUI に設定済みの MCP サーバーがサブエージェントでも直接利用可能になりました（mcpo 経由不要）（[#6](https://github.com/Skyzi000/open-webui-extensions/issues/6)）。

**[Parallel Tools](tools/parallel_tools.py)** ([openwebui.com](https://openwebui.com/posts/parallel_tools_1d44cfce)) - Open WebUI 公式の [Community Newsletter, March 17th 2026](https://openwebui.com/blog/community-newsletter-march-17th-2026) で "Editor's Picks" ツールの1つとして紹介されました。

複数の独立したツール呼び出しを並列実行して処理を高速化します。

## Tools

| ツール | 説明 |
| ------ | ---- |
| [**Sub Agent**](tools/sub_agent.py) | 自律的なサブエージェントにタスクを委譲し、コンテキスト消費を抑制 |
| [**Parallel Tools**](tools/parallel_tools.py) | 複数のツール呼び出しを並列実行して高速化（※強力なフラッグシップモデルでないと正常に呼び出せないことが多いので注意） |
| [**Multi Model Council**](tools/multi_model_council.py) | 複数のモデルによる評議会で多数決を実施 |
| [**User Location**](tools/user_location.py) | ブラウザのGeolocation APIでユーザーの位置情報を取得 |
| [**Universal File Generator (Pandoc)**](tools/universal_file_generator_pandoc.py) | Pandocを利用して様々な形式のファイルを生成 ※現在のところ、自分以外が使用することを想定していません |

## Filters

| フィルター | 説明 |
| ---------- | ---- |
| [**Token Usage Display**](functions/filter/token_usage_display.py) | トークン使用量をコンテキストウィンドウの割合で表示（LiteLLM Proxyが必要） |
| [**Current DateTime Injector**](functions/filter/current_datetime_injector.py) | 現在日時をシステムプロンプトに注入（OpenAIプロンプトキャッシュ活用のためFilter化） |
| [**User Info Injector**](functions/filter/user_info_injector.py) | ユーザー情報をシステムプロンプトに注入（同上） |
| [**Full Context Mode Toggle**](functions/filter/full_context_mode_toggle.py) | チャット単位でフルコンテキストモードを一括切り替え |

## Graphiti Memory（サブモジュール）

[Graphiti](https://github.com/getzep/graphiti)を利用したナレッジグラフベースのメモリ拡張機能です。

別リポジトリ [open-webui-graphiti-memory](https://github.com/Skyzi000/open-webui-graphiti-memory) で管理しています。このリポジトリでは `graphiti/` サブモジュールから参照しています。

- Filter: [graphiti_memory.py](https://github.com/Skyzi000/open-webui-graphiti-memory/blob/main/functions/filter/graphiti_memory.py)
- Tool: [graphiti_memory_manage.py](https://github.com/Skyzi000/open-webui-graphiti-memory/blob/main/tools/graphiti_memory_manage.py)
- Action: [add_graphiti_memory_action.py](https://github.com/Skyzi000/open-webui-graphiti-memory/blob/main/functions/action/add_graphiti_memory_action.py)

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

MIT License
