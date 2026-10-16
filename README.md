# code-chat

Search code by describing it out loud, from Neovim, the terminal or any editor.

Start it and talk. While you speak, the chunks of the repository that best match what you've said so far appear and re-rank every ¾ second, with the matched code shown and highlighted. Stop talking and listening ends; then pick a match.

There's no transcription. [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) embeds your voice with its audio encoder and compares it directly against the code, which it embedded as text. It's the same pattern as the live voice search on [benomahony.com](https://benomahony.com).

Needs [uv](https://docs.astral.sh/uv/), a microphone and, for speed, Apple silicon (the model runs on the GPU through MPS; elsewhere it runs on the CPU).

## Neovim

Needs Neovim 0.11+. While you speak, the best-matching chunks appear as a stack of live, syntax-highlighted code cards beside a waveform, and the window you came from follows the top match, with it highlighted.

```lua
vim.pack.add({ "/Users/benomahony/Code/open_source/code-chat" })
vim.keymap.set("n", "<leader>sv", "<cmd>CodeChat<cr>", { desc = "[S]earch code by [V]oice" })
```

`:CodeChat` opens the cards and starts listening. Describe the code, e.g. "where we retry the download when YouTube refuses". Listening stops after a 2.5 s pause, or after 8 s if you don't say anything.

| Key | Does |
| --- | --- |
| `j` / `k` (or `<C-n>` / `<C-p>`, `<Tab>`) | Move between matches; the main window follows |
| `1`–`9` | Pick a visible card |
| `<CR>` | Go to the match, with every match in the quickfix list (`]q` / `[q` step through them) |
| `<Space>` | Speak again |
| `<Esc>` / `q` | Close and go back to where you were |

Matches stay highlighted after `<CR>` until `:CodeChatClear`. `<C-o>` returns to where you started.

When Neovim opens in a git repository, or `:cd`s into one, code-chat loads the model and indexes it in the background, so the first search doesn't wait. Only what changed since last time is embedded.

`vim.pack` installs a local path by cloning it, so changes here reach Neovim once they're committed and you run `vim.pack.update()`.

`require("code-chat").setup({ cards = 4, width = 0.5, context = 3, index_on_open = true })` changes how many cards show at once, the share of the screen they take, how many lines show above a match in the main window, and whether to index on opening a project.

## Terminal

```sh
uv tool install --editable /Users/benomahony/Code/open_source/code-chat
```

`code-chat` listens in the repository you're in, showing the matches live as syntax-highlighted cards beside a waveform. When you stop talking, `↑`/`↓` (or `j`/`k`) choose one, `<Enter>` prints it as `path:line`, `<Space>` speaks again and `<Esc>` quits.

The live view draws on stderr, so stdout carries only the result:

```sh
nvim $(code-chat)              # or vim, hx, zed: open the match you chose
code -g $(code-chat)           # VS Code, Cursor and the like
code-chat --open               # open it in $VISUAL / $EDITOR at the right line
code-chat --list               # every match, as path:line:col: text
vim -q <(code-chat --list)     # ...into Vim's quickfix list
code-chat ~/Code/some/repo     # search another repository
```

It exits 1, printing nothing on stdout, when you cancel or nothing was heard, and says why on stderr.

The model stays loaded between searches in a background daemon, which the first search starts: that one waits several seconds for EmbeddingGemma 2 to load, and later ones start listening straight away. The daemon serves one search at a time over a socket in `~/.cache/code-chat/`, logs to `daemon.log` there, and exits after 30 minutes without a search. It restarts itself when code-chat's code changes, so an editable install never runs stale code. `code-chat stop` stops it sooner.

## Any editor

`--list` prints matches in the `path:line:col: text` format grep and compilers use, which most editors can jump through: Vim's quickfix list, Emacs' `grep-mode` and `compilation-mode`, and the clickable paths in VS Code's terminal.

For a live integration like the Neovim picker, run `code-chat serve [ROOT]` and talk to it in JSON, one message per line, over stdin and stdout. Or connect to the daemon's socket, `~/.cache/code-chat/daemon.sock`, to share its loaded model: open with `{"type": "open", "root": "/abs/repo", "fingerprint": "..."}`, wait for `{"type": "opened"}` (a `{"type": "restarting"}` means the daemon is older than your code; connect again), then speak the same protocol. `src/code_chat/daemon.py` has the client side.

| Direction | Message |
| --- | --- |
| In | `{"type": "listen"}`, `{"type": "stop"}` |
| In | `{"type": "update", "file": "/abs/path"}` after a save, to re-index that file |
| In | `{"type": "exit"}`, or close stdin |
| Out | `{"type": "status", "text": "Loading EmbeddingGemma 2…"}` while starting |
| Out | `{"type": "ready", "chunks": 1234}` |
| Out | `{"type": "listening", "on": true}` and, about 20 times a second, `{"type": "level", "value": 0.7}` |
| Out | `{"type": "results", "final": false, "seconds": 2.2, "hits": [...]}` on each re-rank; `final` once listening ends |

Each hit has `file` (absolute), `path` (relative to the repository), `start`, `end` and `end_col` (1-based lines, inclusive), `symbols`, a one-line `preview` and its `score`. `lua/code-chat/init.lua` is a complete client.

## How it works

- **Index.** Files come from `git ls-files` (tracked and untracked, minus ignored). [tree-sitter](https://github.com/kreuzberg-dev/tree-sitter-language-pack) splits each into chunks of about a function or two, with 40-line windows for files it can't parse. Each chunk is embedded as `title: <path> | text: <code>`. Vectors are cached by content in `~/.cache/code-chat/`, and a file is re-embedded when the editor reports a save.
- **Listening.** The microphone streams 16 kHz mono. Speech is whatever stands well clear of the room's background level. Once you've said 1.5 s, what you've said so far is embedded again after each further ¾ s, and chunks rank by cosine to it. That's averaged with earlier updates, so results only move when they're consistently better.

## Development

The Python is held to [zanity](https://github.com/benomahony/zanity) with every rule on (`zanity.toml`), and to [basedpyright](https://docs.basedpyright.com) in strict mode, with no warnings:

```sh
zanity check . --strict
uv run basedpyright
```

`.github/workflows/check.yml` runs both on every change and daily. `typings/` holds type stubs for sounddevice, which ships none. `zanity check . --strict --infer` also asks a language model to judge what structure can't, such as error messages that don't say what to do.
