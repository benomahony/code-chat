if vim.g.loaded_code_chat then
  return
end
vim.g.loaded_code_chat = true

vim.api.nvim_create_user_command("CodeChat", function()
  require("code-chat").pick()
end, { desc = "Search code by describing it out loud" })

require("code-chat").setup()
