-- Search code by describing it out loud. The Python daemon
-- (src/code_chat/server.py) embeds the repository with EmbeddingGemma 2 and
-- re-ranks its chunks against your voice while you talk. Here, the best
-- chunks show live as a stack of code cards beside a waveform, and the window
-- you came from follows the top match with it highlighted.

local M = {}

local config = {
  -- Code cards shown at once; j/k scroll through the rest.
  cards = 4,
  -- Share of the screen's width the cards take.
  width = 0.5,
  -- Lines of context kept above a match in the main window.
  context = 3,
}

local ns = vim.api.nvim_create_namespace("code-chat")
local plugin_root = vim.fn.fnamemodify(debug.getinfo(1, "S").source:sub(2), ":h:h:h")

local server = { job = nil, root = nil, ready = false, status = "" }
-- The current search. `moved` is set once you pick a hit yourself, so live
-- updates stop resetting the selection to the top one.
local state = { hits = {}, selected = 1, first = 1, listening = false, level = 0, final = false, moved = false }
local ui

local function set_highlights()
  local set = function(name, value)
    value.default = true
    vim.api.nvim_set_hl(0, name, value)
  end
  set("CodeChatSelected", { link = "Visual" })
  set("CodeChatMatch", { link = "CursorLine" })
  set("CodeChatRank", { link = "DiagnosticVirtualTextInfo" })
  set("CodeChatSign", { link = "DiagnosticInfo" })
  set("CodeChatMeter", { link = "DiagnosticOk" })
  set("CodeChatMeterIdle", { link = "NonText" })
  set("CodeChatStatus", { link = "Comment" })
  set("CodeChatBorder", { link = "FloatBorder" })
  set("CodeChatBorderSelected", { link = "DiagnosticInfo" })
  set("CodeChatFlash", { link = "DiagnosticWarn" })
  set("CodeChatScore", { link = "DiagnosticOk" })
  set("CodeChatScoreEmpty", { link = "NonText" })
end

local function valid(win)
  return win and vim.api.nvim_win_is_valid(win)
end

local function key(hit)
  return hit.file .. ":" .. hit.start
end

-- --- Highlights in code buffers ---------------------------------------------

-- Every hit in `buf` gets a soft background and its rank; the selected one
-- stands out. They show in the main window and the cards alike.
local function mark(buf)
  if not vim.api.nvim_buf_is_valid(buf) then
    return
  end
  vim.api.nvim_buf_clear_namespace(buf, ns, 0, -1)
  local name = vim.api.nvim_buf_get_name(buf)
  local count = vim.api.nvim_buf_line_count(buf)
  for i, hit in ipairs(state.hits) do
    if hit.file == name and hit.start <= count then
      local selected = i == state.selected
      vim.api.nvim_buf_set_extmark(buf, ns, hit.start - 1, 0, {
        end_row = math.min(hit["end"], count),
        hl_group = selected and "CodeChatSelected" or "CodeChatMatch",
        hl_eol = true,
        priority = selected and 200 or 150,
        sign_text = selected and "▶" or "▍",
        sign_hl_group = "CodeChatSign",
        virt_text = { { (" #%d "):format(i), "CodeChatRank" } },
        virt_text_pos = "right_align",
      })
    end
  end
end

local function mark_all()
  for _, buf in ipairs(vim.api.nvim_list_bufs()) do
    if vim.api.nvim_buf_is_loaded(buf) then
      mark(buf)
    end
  end
end

function M.clear()
  state.hits = {}
  for _, buf in ipairs(vim.api.nvim_list_bufs()) do
    if vim.api.nvim_buf_is_valid(buf) then
      vim.api.nvim_buf_clear_namespace(buf, ns, 0, -1)
    end
  end
end

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
  server.job = vim.fn.jobstart({ "uv", "run", "--quiet", "--directory", plugin_root, "code-chat", dir }, {
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

-- --- Waveform -------------------------------------------------------------------

-- Two rows of bars driven by the microphone level, tallest in the middle and
-- each wobbling at its own pace, so it reads as a voice. Once listening
-- stops they settle into a row of dots.
local BLOCKS = { " ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█" }
local heights = {}

local function waveform(width)
  local t = vim.uv.hrtime() / 1e9
  local top, bottom = {}, {}
  for i = 1, width do
    local x = (i - (width + 1) / 2) / (width * 0.28)
    local envelope = 0.15 + 0.85 * math.exp(-x * x)
    local wobble = 0.55 + 0.45 * math.sin(t * (5 + (i * 7) % 5) + i * 2.399)
    local target = state.listening and state.level * envelope * wobble or 0
    local h = heights[i] or 0
    -- Rise quickly, fall back gently.
    h = h + (target - h) * (target > h and 0.5 or 0.15)
    heights[i] = h
    local eighths = math.floor(h * 16 + 0.5)
    if not state.listening and eighths == 0 then
      top[i], bottom[i] = " ", "·"
    else
      top[i] = BLOCKS[math.max(0, math.min(8, eighths - 8)) + 1]
      bottom[i] = BLOCKS[math.max(1, math.min(8, eighths)) + 1]
    end
  end
  return table.concat(top), table.concat(bottom)
end

local function status_text()
  if not server.ready then
    return server.status
  end
  if state.listening then
    return #state.hits > 0 and ("Listening… %.1f s heard"):format(state.seconds or 0) or "Listening… describe the code"
  end
  if state.final and #state.hits > 0 then
    return ("Closest to what you said · %d/%d"):format(state.selected, #state.hits)
  end
  return state.status or server.status
end

local function draw_header()
  if not ui or not vim.api.nvim_buf_is_valid(ui.header_buf) then
    return
  end
  local top, bottom = waveform(ui.width)
  local lines = { top, bottom, " " .. status_text() }
  vim.api.nvim_buf_set_lines(ui.header_buf, 0, -1, false, lines)
  vim.api.nvim_buf_clear_namespace(ui.header_buf, ns, 0, -1)
  local meter = state.listening and "CodeChatMeter" or "CodeChatMeterIdle"
  for row = 0, 1 do
    vim.api.nvim_buf_set_extmark(ui.header_buf, ns, row, 0, { end_col = #lines[row + 1], hl_group = meter })
  end
  vim.api.nvim_buf_set_extmark(ui.header_buf, ns, 2, 0, { end_col = #lines[3], hl_group = "CodeChatStatus" })
end

-- --- Code cards -------------------------------------------------------------------

-- How well a hit matches next to the best one, as a bar for the card's footer.
local function score_bar(hit)
  local best = state.hits[1].score
  local worst = state.hits[#state.hits].score
  local share = best > worst and (hit.score - worst) / (best - worst) or 1
  local filled = math.floor(share * 10 + 0.5)
  return {
    { " " .. ("▰"):rep(filled), "CodeChatScore" },
    { ("▱"):rep(10 - filled), "CodeChatScoreEmpty" },
    { (" %.3f "):format(hit.score), "CodeChatStatus" },
  }
end

local function load(hit)
  local buf = vim.fn.bufadd(hit.file)
  vim.fn.bufload(buf)
  return buf
end

local function show_at(win, buf, line, context)
  if vim.api.nvim_win_get_buf(win) ~= buf then
    vim.api.nvim_win_set_buf(win, buf)
  end
  line = math.min(line, vim.api.nvim_buf_line_count(buf))
  vim.api.nvim_win_set_cursor(win, { line, 0 })
  vim.api.nvim_win_call(win, function()
    vim.fn.winrestview({ topline = math.max(1, line - context) })
  end)
end

local function draw_cards()
  if not ui then
    return
  end
  -- Keep the selection in view.
  local n = #ui.cards
  if state.selected < state.first then
    state.first = state.selected
  elseif state.selected > state.first + n - 1 then
    state.first = state.selected - n + 1
  end
  state.first = math.max(1, math.min(state.first, #state.hits - n + 1))
  local now = vim.uv.now()
  for slot, card in ipairs(ui.cards) do
    local i = state.first + slot - 1
    local hit = state.hits[i]
    if valid(card.win) then
      -- Cards stay hidden until there's a match to show in them.
      vim.api.nvim_win_set_config(card.win, { hide = not hit })
      if not hit then
        card.key = nil
      else
        if card.key ~= key(hit) then
          -- A new match arrived in this place: flash it.
          card.key = key(hit)
          card.flash = now + 400
        end
        show_at(card.win, load(hit), hit.start, 0)
        -- Switching buffers brings back that buffer's window options.
        local wo = vim.wo[card.win]
        wo.number, wo.relativenumber, wo.signcolumn, wo.wrap, wo.foldenable = true, false, "no", false, false
        local symbols = #hit.symbols > 0 and ("  " .. table.concat(hit.symbols, ", ")) or ""
        local selected = i == state.selected
        vim.api.nvim_win_set_config(card.win, {
          title = {
            { (" #%d "):format(i), selected and "CodeChatBorderSelected" or "CodeChatRank" },
            { (" %s:%d%s "):format(hit.path, hit.start, symbols), "FloatTitle" },
          },
          footer = score_bar(hit),
          footer_pos = "right",
        })
        local border = card.flash > now and "CodeChatFlash"
          or selected and "CodeChatBorderSelected"
          or "CodeChatBorder"
        vim.wo[card.win].winhighlight = "FloatBorder:" .. border
      end
    end
  end
end

-- The main window follows the selected hit.
local function preview()
  local hit = state.hits[state.selected]
  if ui and hit and valid(ui.origin) then
    show_at(ui.origin, load(hit), hit.start, config.context)
  end
end

local function redraw()
  draw_cards()
  preview()
  mark_all()
  draw_header()
end

local function select(i)
  if #state.hits == 0 then
    return
  end
  state.moved = true
  state.selected = ((i - 1) % #state.hits) + 1
  redraw()
end

-- --- Opening and closing ----------------------------------------------------------

local function close()
  if not ui then
    return
  end
  local closing = ui
  ui = nil
  closing.timer:stop()
  closing.timer:close()
  if state.listening then
    send({ type = "stop" })
  end
  for _, win in ipairs(vim.list_extend({ closing.header_win }, vim.tbl_map(function(c)
    return c.win
  end, closing.cards))) do
    if valid(win) then
      vim.api.nvim_win_close(win, true)
    end
  end
  if vim.api.nvim_buf_is_valid(closing.blank) then
    vim.api.nvim_buf_delete(closing.blank, { force = true })
  end
  return closing
end

local function cancel()
  local closed = close()
  M.clear()
  if closed and valid(closed.origin) and vim.api.nvim_buf_is_valid(closed.buf) then
    vim.api.nvim_set_current_win(closed.origin)
    vim.api.nvim_win_set_buf(closed.origin, closed.buf)
    vim.fn.winrestview(closed.view)
  end
end

-- Stay at the selected hit, with every hit in the quickfix list, so ]q and
-- [q step through them, and keep their highlights until :CodeChatClear.
local function accept()
  if not state.hits[state.selected] then
    return cancel()
  end
  local closed = close()
  vim.fn.setqflist({}, " ", {
    title = "Code chat",
    idx = state.selected,
    items = vim.tbl_map(function(h)
      return {
        filename = h.file,
        lnum = h.start,
        end_lnum = h["end"],
        text = (#h.symbols > 0 and table.concat(h.symbols, ", ") .. ": " or "") .. h.preview,
      }
    end, state.hits),
  })
  if valid(closed.origin) then
    vim.api.nvim_set_current_win(closed.origin)
    vim.bo.buflisted = true
  end
  mark_all()
end

local function listen()
  state.moved = false
  send({ type = "listen" })
end

local function open()
  set_highlights()
  start()
  M.clear()
  state.selected, state.first, state.final, state.moved, state.status = 1, 1, false, false, nil
  heights = {}
  local origin = vim.api.nvim_get_current_win()
  local view = vim.fn.winsaveview()
  vim.cmd("normal! m'") -- So <C-o> comes back here after jumping to a hit.

  local width = math.max(50, math.floor(vim.o.columns * config.width))
  local col = vim.o.columns - width - 2
  local bottom = vim.o.lines - vim.o.cmdheight - 1
  local header_buf = vim.api.nvim_create_buf(false, true)
  vim.bo[header_buf].bufhidden = "wipe"
  local header_win = vim.api.nvim_open_win(header_buf, true, {
    relative = "editor",
    row = 0,
    col = col,
    width = width,
    height = 3,
    border = "rounded",
    style = "minimal",
    title = { { " 󰍬 Code chat ", "FloatTitle" } },
    title_pos = "center",
    footer = { { " j/k move · ⏎ open · ␣ speak again · esc close ", "CodeChatStatus" } },
    footer_pos = "center",
  })

  -- Cards fill the rest of the column, each the same height.
  local top = 5
  local count = config.cards
  local height = math.floor((bottom - top) / count) - 2
  while height < 4 and count > 1 do
    count = count - 1
    height = math.floor((bottom - top) / count) - 2
  end
  local blank = vim.api.nvim_create_buf(false, true)
  local cards = {}
  for slot = 1, count do
    local win = vim.api.nvim_open_win(blank, false, {
      relative = "editor",
      row = top + (slot - 1) * (height + 2),
      col = col,
      width = width,
      height = height,
      border = "rounded",
      style = "minimal",
      focusable = false,
      hide = true,
    })
    cards[slot] = { win = win, flash = 0 }
  end

  ui = {
    origin = origin,
    buf = vim.api.nvim_win_get_buf(origin),
    view = view,
    header_buf = header_buf,
    header_win = header_win,
    blank = blank,
    cards = cards,
    width = width,
    timer = vim.uv.new_timer(),
  }
  -- The waveform and card flashes animate between messages.
  ui.timer:start(0, 33, vim.schedule_wrap(function()
    if ui then
      draw_header()
      for _, card in ipairs(ui.cards) do
        if card.flash > 0 and card.flash <= vim.uv.now() then
          card.flash = 0
          draw_cards()
        end
      end
    end
  end))

  local map = function(keys, fn)
    for _, lhs in ipairs(keys) do
      vim.keymap.set("n", lhs, fn, { buffer = header_buf, nowait = true })
    end
  end
  map({ "j", "<Down>", "<C-n>", "<Tab>" }, function()
    select(state.selected + 1)
  end)
  map({ "k", "<Up>", "<C-p>", "<S-Tab>" }, function()
    select(state.selected - 1)
  end)
  for n = 1, 9 do
    map({ tostring(n) }, function()
      select(state.first + n - 1)
    end)
  end
  map({ "<CR>" }, accept)
  map({ "<Esc>", "q", "<C-c>" }, cancel)
  map({ "<Space>" }, listen)
  vim.api.nvim_create_autocmd("WinLeave", {
    buffer = header_buf,
    once = true,
    callback = function()
      vim.schedule(function()
        if ui and vim.api.nvim_get_current_win() ~= ui.header_win then
          cancel()
        end
      end)
    end,
  })
end

-- Open the cards and start listening.
function M.listen()
  if not ui then
    open()
  end
  listen()
end

-- --- Messages from the daemon -------------------------------------------------

handle = function(message)
  local kind = message.type
  if kind == "status" then
    server.status = message.text
    state.status = message.text
  elseif kind == "ready" then
    server.ready = true
    server.status = ("%d chunks indexed"):format(message.chunks)
  elseif kind == "listening" then
    state.listening = message.on
    if message.on then
      state.final = false
    else
      state.level = 0
    end
  elseif kind == "level" then
    state.level = message.value
  elseif kind == "results" and ui then
    local selected = state.hits[state.selected]
    state.hits = message.hits
    state.seconds = message.seconds
    state.final = message.final
    state.selected = 1
    -- Keep a hit you picked selected while results keep moving.
    if state.moved and selected then
      for i, hit in ipairs(state.hits) do
        if key(hit) == key(selected) then
          state.selected = i
        end
      end
    end
    redraw()
  end
  draw_header()
end

-- --- Setup ----------------------------------------------------------------------

function M.setup(opts)
  config = vim.tbl_extend("force", config, opts or {})
  set_highlights()
  local group = vim.api.nvim_create_augroup("code-chat", { clear = true })
  vim.api.nvim_create_autocmd("ColorScheme", { group = group, callback = set_highlights })
  -- Hits stay highlighted in files opened later, e.g. from the quickfix list.
  vim.api.nvim_create_autocmd("BufWinEnter", {
    group = group,
    callback = function(args)
      if #state.hits > 0 and not ui then
        mark(args.buf)
      end
    end,
  })
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
