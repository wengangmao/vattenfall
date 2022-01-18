# Logbook for Vattenfall project

This log book is written as a Markdown format based on [JupyterBook](https://jupyterbook.org/start/your-first-book.html).

In order to build the markdown into html pages, you need to download:
- jupter book (build html files from markdown files)
- ghp-import  (upload html files onto Github to publish the website)

Run the following steps to compile the Jupter book:

> 1. Go to upper level directory containing the "vattenfall" folder
> 2. Build the book by run "jupter-book build vattenfall"
> 3. Enter into the "vattenfall" folder
> 4. Publish the html pages by run "ghp-import -n -p -f _build/html"

Or, alternatively, an easier approach to build as follows:

> 1. Go to the "vattenfall" folder
> 2. Build the book by run "jb build ."
> 3. Publish the html pages by run "ghp-import -n -p -f _build/html"

Then you should be able to access the [Vattenfall_project logbook](https://wengangmao.github.io/vattenfall/contents/home.html)!