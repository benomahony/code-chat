# code-chat

Search code by describing it out loud, from Neovim, the terminal or any editor.

Start it and talk. While you speak, the chunks of the repository that best match what you've said so far appear and re-rank every ¾ second, with the matched code shown and highlighted. Stop talking and listening ends; then pick a match.

There's no transcription. [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) embeds your voice with its audio encoder and compares it directly against the code, which it embedded as text. It's the same pattern as the live voice search on [benomahony.com](https://benomahony.com).

Needs [uv](https://docs.astral.sh/uv/), a microphone and, for speed, Apple silicon (the model runs on the GPU through MPS; elsewhere it runs on the CPU).

## Neovim

A [Snacks](https://github.com/folke/snacks.nvim) picker: the list re-ranks live as you talk, the preview shows the matched chunk highlighted, and the title shows the microphone's waveform.

```lua
vim.pack.add({ "https://github.com/folke/snacks.nvim", "/Users/benomahony/Code/open_source/code-chat" })
vim.keymap.set("n", "<leader>sv", "<cmd>CodeChat<cr>", { desc = "Search code by [V]oice" })
```

`:CodeChat` opens the picker and starts listening. Listening stops after a 2.5 s pause, or after 8 s if you don't say anything. Then it's an ordinary Snacks picker: `<CR>` opens the match, `<C-q>` sends every match to the quickfix list, `<C-v>` / `<C-s>` open in a split, and typing filters the matches. `<C-Space>` speaks again.

`vim.pack` installs a local path by cloning it, so changes here reach Neovim once they're committed and you run `vim.pack.update()`. The first search sets up the Python environment, loads the model and embeds the repository; later runs only embed what changed. To load the model when Neovim starts instead:

```lua
vim.api.nvim_create_autocmd("VimEnter", { callback = function() require("code-chat").start() end })
```

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

It exits 1, printing nothing, when you cancel or nothing was heard.

## Any editor

`--list` prints matches in the `path:line:col: text` format grep and compilers use, which most editors can jump through: Vim's quickfix list, Emacs' `grep-mode` and `compilation-mode`, and the clickable paths in VS Code's terminal.

For a live integration like the Neovim picker, run `code-chat serve [ROOT]` and talk to it in JSON, one message per line, over stdin and stdout:

| Direction | Message |
| --- | --- |
| In | `{"type": "listen"}`, `{"type": "stop"}` |
| In | `{"type": "update", "file": "/abs/path"}` after a save, to re-index that file |
| In | `{"type": "exit"}`, or close stdin |
| Out | `{"type": "status", "text": "Loading EmbeddingGemma 2…"}` while starting |
| Out | `{"type": "ready", "chunks": 1234}` |
| Out | `{"type": "listening", "on": true}` and, about 20 times a second, `{"type": "level", "value": 0.7}` |
| Out | `{"type": "results", "final": false, "seconds": 2.2, "hits": [...]}` on each re-rank; `final` once listening ends |

Each hit has `file` (absolute), `path` (relative to the repository), `start`, `end` and `end_col` (1-based lines, inclusive), `symbols`, a one-line `preview` and its `score`. `lua/code-chat/init.lua` is a complete client in under 250 lines.

## How it works

- **Index.** Files come from `git ls-files` (tracked and untracked, minus ignored). [tree-sitter](https://github.com/kreuzberg-dev/tree-sitter-language-pack) splits each into chunks of about a function or two, with 40-line windows for files it can't parse. Each chunk is embedded as `title: <path> | text: <code>`. Vectors are cached by content in `~/.cache/code-chat/`, and a file is re-embedded when the editor reports a save.
- **Listening.** The microphone streams 16 kHz mono. Speech is whatever stands well clear of the room's background level. Once you've said 1.5 s, what you've said so far is embedded again after each further ¾ s, and chunks rank by cosine to it. That's averaged with earlier updates, so results only move when they're consistently better.

## Development

The Python is held to [zanity](https://github.com/benomahony/zanity) with every rule on (`zanity.toml`), strict:

```sh
zanity check . --strict
```
