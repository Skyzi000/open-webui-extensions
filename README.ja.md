# Open WebUI Extensions

このリポジトリは、Open WebUI用のツールとフィルターを含んでいます。

## 🌟 ハイライト

**[Sub Agent Tool](tools/sub_agent.py)** ([openwebui.com](https://openwebui.com/posts/sub_agent_7bfeb0b7)) - openwebui.com で **Upvote数1位**、ダウンロード数 **2万+** を達成！ Open WebUI 公式の [Community Newsletter, January 28th 2026](https://openwebui.com/blog/newsletter-january-28-2026) で "This Week's Most Useful" ツールの1つとして紹介されました。

ツール呼び出しが多いタスクをサブエージェントに委譲し、メインの会話コンテキストをクリーンに保つツールです。Open WebUI v0.7 以降のビルトインツール（Web検索、メモリ、ナレッジベース等）を最大限に活用できます。

### What's New

- **v0.6** — サブエージェント内の自動コンテキスト圧縮と、大きなツール結果のプレビュー・読み戻しに対応。最低対応版を Open WebUI 0.9.6 へ引き上げました（従来 0.7.0）。[詳細](#サブエージェントのコンテキスト圧縮)
- **v0.5** — Open WebUI に設定済みの MCP サーバーがサブエージェントでも直接利用可能になりました（mcpo 経由不要）（[#6](https://github.com/Skyzi000/open-webui-extensions/issues/6)）
- **v0.4.5** — Open Terminal のツール（Open WebUI v0.8.6+）が自動的にサブエージェントに転送されます
- **v0.4** — Open WebUI v0.8 で導入されたスキルが自動的にサブエージェントへ伝播されます（実験的機能）
- **v0.3** — `run_parallel_sub_agents` によるサブエージェントの並列実行をネイティブサポート

フルチェンジログ: [sub_agent.py のコミット履歴](https://github.com/Skyzi000/open-webui-extensions/commits/main/tools/sub_agent.py)

> [!TIP]
> 並列実行で問題が発生する場合（検索APIのレートリミット等）は、Valvesの `MAX_PARALLEL_AGENTS` を下げるか、`run_parallel_sub_agents` メソッドをコメントアウトして無効化してください。

### サブエージェントのコンテキスト圧縮

ツール呼び出しを重ねると、サブエージェント自身のコンテキストにも履歴や結果が蓄積されます。v0.6 では、次の 2 つの仕組みでモデルへ送るコンテキストを抑え、長い調査や作業を続けやすくします。どちらもデフォルトで有効で、Valves から個別に設定できます。

- **履歴の自動圧縮** — 推定入力トークン数が閾値（既定: 80,000）に達すると、古いラウンドを要約し、依頼内容と直近のラウンドは残します。要約時も、既定では同じモデルを使い、送信済みの冒頭部分やツール定義を再利用することで、プロンプトキャッシュをできるだけ維持しやすいよう配慮しています。
- **大きなツール結果のプレビュー** — 大きな結果（既定では 10,000 トークン以上、または 64 KiB 超）は、全文の代わりに先頭と末尾の短いプレビューを送ります。元の全文は同じタスクの実行中に保持され、サブエージェントは `agent_ref_exec` ツールで必要な行や範囲を読み戻せます（`head`、`tail`、`sed -n`、`grep` などのコマンド形式）。

Open WebUI 標準のチャット履歴圧縮は、このツールが内部で回すサブエージェントのループには適用されません。この機能は、その内部履歴を対象としています。

| Valve | 役割 |
| ----- | ---- |
| `ENABLE_CONTEXT_COMPACTION` | 履歴の自動圧縮の有効／無効（既定: 有効） |
| `CONTEXT_COMPACTION_TOKEN_THRESHOLD` | 圧縮を始める推定入力トークン数（既定: 80,000） |
| `COMPACTION_SUMMARY_MODEL` | 要約に使うモデル。空ならサブエージェントと同じモデル（推奨） |
| `LARGE_TOOL_RESULT_MODE` | `ref_exec`（プレビュー＋読み戻し、既定）/ `truncate`（中間を省略、読み戻しなし）/ `raw`（そのまま送る） |
| `LARGE_TOOL_RESULT_THRESHOLD_TOKENS` | 大きな結果とみなすトークン数の閾値（既定: 10,000） |
| `MAX_ITERATIONS` | 反復上限（既定: 50、従来: 10。0 で無制限） |

**[Parallel Tools](tools/parallel_tools.py)** ([openwebui.com](https://openwebui.com/posts/parallel_tools_1d44cfce)) - Open WebUI 公式の [Community Newsletter, March 17th 2026](https://openwebui.com/blog/community-newsletter-march-17th-2026) で "Editor's Picks" ツールの1つとして紹介されました。

複数の独立したツール呼び出しを並列実行して処理を高速化します。

## Tools

| ツール | 説明 |
| ------ | ---- |
| [**Sub Agent**](tools/sub_agent.py) | 自律的なサブエージェントにタスクを委譲し、コンテキスト消費を抑制 |
| [**Parallel Tools**](tools/parallel_tools.py) | 複数のツール呼び出しを並列実行して高速化（※強力なフラッグシップモデルでないと正常に呼び出せないことが多いので注意） |
| [**Multi Model Council**](tools/multi_model_council.py) | 複数のモデルによる評議会で多数決を実施 |
| [**LLM Review**](tools/llm_review.py) | 創造性の発散を保つ多ペルソナ創作ライティング — ペルソナごとに独立起草・相互レビュー・改訂を行い、マージせず個性の異なるドラフトを複数返す（arXiv:2601.08003 を参考に独自実装） |
| [**User Location**](tools/user_location.py) | ブラウザのGeolocation APIでユーザーの位置情報を取得 |
| [**Universal File Generator (Pandoc)**](tools/universal_file_generator_pandoc.py) | Pandocを利用して様々な形式のファイルを生成 ※現在のところ、自分以外が使用することを想定していません |

## Filters

| フィルター | 説明 |
| ---------- | ---- |
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
