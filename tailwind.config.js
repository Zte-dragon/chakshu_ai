/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './chakshu_ai/templates/**/*.html',
    './node_modules/flowbite/**/*.js'
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('flowbite/plugin')
  ],
}
