# code-chat

Search code by describing it out loud, from Neovim.

Press a key and start talking. While you speak, the best-matching chunks of the repository appear as a stack of live, syntax-highlighted code cards beside a waveform, re-ranking every ¾ second. The window you came from follows the top match, with it highlighted. Stop talking and listening ends; then move between matches, open one, or speak again.

There's no transcription. [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) embeds your voice with its audio encoder and compares it directly against the code, which it embedded as text. It's the same pattern as the live voice search on [benomahony.com](https://benomahony.com).

## Install

Needs [uv](https://docs.astral.sh/uv/), Neovim 0.11+, a microphone and, for speed, Apple silicon (the model runs on the GPU through MPS; elsewhere it runs on the CPU).

```lua
vim.pack.add({ "/Users/benomahony/Code/open_source/code-chat" })
vim.keymap.set("n", "<leader>sv", "<cmd>CodeChat<cr>", { desc = "[S]earch code by [V]oice" })
```

The first search downloads nothing if `google/embeddinggemma-2` is already in the Hugging Face cache. Otherwise it downloads the model, then embeds the repository. Later runs only embed what changed.

## Use

`:CodeChat` opens the cards and starts listening. Describe the code, e.g. "where we retry the download when YouTube refuses". Listening stops after a 2.5 s pause, or after 8 s if you don't say anything.

| Key | Does |
| --- | --- |
| `j` / `k` (or `<C-n>` / `<C-p>`, `<Tab>`) | Move between matches; the main window follows |
| `1`–`9` | Pick a visible card |
| `<CR>` | Go to the match, with every match in the quickfix list (`]q` / `[q` step through them) |
| `<Space>` | Speak again |
| `<Esc>` / `q` | Close and go back to where you were |

Matches stay highlighted after `<CR>` until `:CodeChatClear`. `<C-o>` returns to where you started.

Optionally, start the model when Neovim opens, so the first search doesn't wait for it:

```lua
vim.api.nvim_create_autocmd("VimEnter", { callback = function() require("code-chat").start() end })
```

`require("code-chat").setup({ cards = 4, width = 0.5, context = 3 })` changes how many cards show at once, the share of the screen they take, and how many lines show above a match in the main window.

## How it works

- **Index.** Files come from `git ls-files` (tracked and untracked, minus ignored). [tree-sitter](https://github.com/kreuzberg-dev/tree-sitter-language-pack) splits each into chunks of about a function or two, with 40-line windows for files it can't parse. Each chunk is embedded as `title: <path> | text: <code>`. Vectors are cached by content in `~/.cache/code-chat/`, and a file is re-embedded when you save it.
- **Listening.** The microphone streams 16 kHz mono. Speech is whatever stands well clear of the room's background level. Once you've said 1.5 s, what you've said so far is embedded again after each further ¾ s, and chunks rank by cosine to it. That's averaged with earlier updates, so results only move when they're consistently better.
- **Neovim.** `lua/code-chat/init.lua` starts the daemon (`src/code_chat/server.py`) with `uv run` and talks to it in JSON lines over stdin and stdout.
