-- Search code by describing it out loud, in a Snacks picker. The Python
-- daemon (src/code_chat/server.py) embeds the repository with EmbeddingGemma
-- 2 and re-ranks its chunks against your voice while you talk; the picker's
-- list re-ranks with it, the preview shows the matched chunk highlighted, and
-- the title shows what the microphone hears.

local M = {}

local plugin_root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h:h")

local server = { job = nil, root = nil, ready = false, status = "" }
-- The open picker and what it shows.
local state = { picker = nil, hits = {}, listening = false, level = 0, seconds = 0, final = false }

-- --- Daemon -------------------------------------------------------------------

local handle

local function root()
  return vim.fs.root(0, ".git") or vim.fn.getcwd()
end

local function start()
  local dir = root()
  if server.job and server.root == dir then
    return
  end
  if server.job then
    vim.fn.jobstop(server.job)
  end
  server.root, server.ready, server.status = dir, false, "Starting…"
  local partial = ""
  local errors = {}
  server.job = vim.fn.jobstart({ "uv", "run", "--quiet", "--directory", plugin_root, "code-chat", "serve", dir }, {
    on_stdout = function(_, data)
      data[1] = partial .. data[1]
      partial = table.remove(data)
      for _, line in ipairs(data) do
        if line ~= "" then
          handle(vim.json.decode(line))
        end
      end
    end,
    on_stderr = function(_, data)
      vim.list_extend(errors, data)
      errors = vim.list_slice(errors, math.max(1, #errors - 20))
    end,
    on_exit = function(job, code)
      if server.job ~= job then
        return
      end
      server.job, server.ready = nil, false
      if code ~= 0 and code ~= 143 then
        vim.notify("code-chat stopped (" .. code .. "):\n" .. table.concat(errors, "\n"), vim.log.levels.ERROR)
      end
    end,
  })
end

local function send(message)
  start()
  vim.fn.chansend(server.job, vim.json.encode(message) .. "\n")
end

-- --- Title ------------------------------------------------------------------------

-- A row of bars driven by the microphone level, tallest in the middle and
-- each wobbling at its own pace, so it reads as a voice.
local BLOCKS = { "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█" }
local WIDTH = 12
local heights = {}

local function waveform()
  local t = vim.uv.hrtime() / 1e9
  local bars = {}
  for i = 1, WIDTH do
    local x = (i - (WIDTH + 1) / 2) / (WIDTH * 0.3)
    local wobble = 0.55 + 0.45 * math.sin(t * (5 + (i * 7) % 5) + i * 2.399)
    local target = state.listening and state.level * (0.15 + 0.85 * math.exp(-x * x)) * wobble or 0
    local h = heights[i] or 0
    -- Rise quickly, fall back gently.
    heights[i] = h + (target - h) * (target > h and 0.5 or 0.15)
    bars[i] = BLOCKS[math.max(1, math.min(#BLOCKS, math.floor(heights[i] * #BLOCKS) + 1))]
  end
  return table.concat(bars)
end

local function title()
  local text
  if not server.ready then
    text = server.status
  elseif state.listening then
    text = state.seconds > 0 and ("Listening… %.1f s"):format(state.seconds) or "Listening… describe the code"
  elseif state.final then
    text = "Closest to what you said  (<C-Space> speak again)"
  else
    text = server.status
  end
  return "󰍬 " .. waveform() .. "  " .. text
end

local function redraw_title()
  local picker = state.picker
  if picker and not picker.closed then
    picker.title = title()
    picker:update_titles()
  end
end

-- --- Picker -----------------------------------------------------------------------

local function items()
  return vim.tbl_map(function(hit)
    local symbols = table.concat(hit.symbols, " ")
    return {
      file = hit.file,
      -- The preview highlights the chunk from pos to end_pos.
      pos = { hit.start, 0 },
      end_pos = { hit["end"], hit.end_col },
      line = hit.preview,
      text = ("%s %s %s"):format(hit.path, symbols, hit.preview),
    }
  end, state.hits)
end

local function listen()
  state.final, state.seconds, heights = false, 0, {}
  send({ type = "listen" })
end

-- Open the picker and start listening.
function M.pick()
  local ok, Snacks = pcall(require, "snacks")
  if not ok then
    return vim.notify("code-chat needs snacks.nvim for its picker", vim.log.levels.ERROR)
  end
  start()
  state.hits, state.listening, state.level, state.seconds = {}, false, 0, 0
  local timer = vim.uv.new_timer()
  state.picker = Snacks.picker.pick({
    source = "code_chat",
    title = title(),
    finder = items,
    format = "file",
    preview = "file",
    -- It starts empty and fills as you talk.
    show_empty = true,
    -- Keep the daemon's ranking until you type to filter it.
    matcher = { sort_empty = false },
    actions = { code_chat_listen = listen },
    win = {
      input = { keys = { ["<c-space>"] = { "code_chat_listen", mode = { "i", "n" }, desc = "Speak again" } } },
      list = { keys = { ["<c-space>"] = { "code_chat_listen", desc = "Speak again" } } },
    },
    on_close = function()
      timer:stop()
      timer:close()
      state.picker = nil
      if state.listening then
        send({ type = "stop" })
      end
    end,
  })
  -- The waveform animates between messages.
  timer:start(0, 50, vim.schedule_wrap(redraw_title))
  listen()
end

-- --- Messages from the daemon -------------------------------------------------

handle = function(message)
  local kind = message.type
  if kind == "status" then
    server.status = message.text
  elseif kind == "ready" then
    server.ready = true
    server.status = ("%d chunks indexed"):format(message.chunks)
  elseif kind == "listening" then
    state.listening = message.on
    if not message.on then
      state.level = 0
    end
  elseif kind == "level" then
    state.level = message.value
  elseif kind == "results" then
    state.hits = message.hits
    state.seconds = message.seconds
    state.final = message.final
    local picker = state.picker
    if picker and not picker.closed then
      picker:find({ refresh = true })
    end
  end
  redraw_title()
end

-- --- Setup ----------------------------------------------------------------------

function M.setup()
  local group = vim.api.nvim_create_augroup("code-chat", { clear = true })
  -- Keep the index current as files are saved.
  vim.api.nvim_create_autocmd("BufWritePost", {
    group = group,
    callback = function(args)
      if server.job and server.root and vim.startswith(args.match, server.root .. "/") then
        send({ type = "update", file = args.match })
      end
    end,
  })
  vim.api.nvim_create_autocmd("VimLeavePre", {
    group = group,
    callback = function()
      if server.job then
        vim.fn.jobstop(server.job)
      end
    end,
  })
end

-- Start the daemon ahead of the first search, so the model is loaded.
M.start = start

return M
