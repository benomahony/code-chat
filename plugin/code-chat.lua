if vim.g.loaded_code_chat then
  return
end
vim.g.loaded_code_chat = true

vim.api.nvim_create_user_command("CodeChat", function()
  require("code-chat").listen()
end, { desc = "Search code by describing it out loud" })
vim.api.nvim_create_user_command("CodeChatClear", function()
  require("code-chat").clear()
end, { desc = "Clear code chat highlights" })

require("code-chat").setup()
