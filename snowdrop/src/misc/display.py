"""Display content of a model file."""
from snowdrop.src.utils.load import read_file_or_url

def model_to_html(filename):
    """
    Display model file.
    
        :param filename: Path to model file
        :type filename: str.
        :returns:  HTML representation of model file
    """

    code,labels = read_file_or_url(filename)

    HTML_TEMPLATE = """<style>
    {}
    </style>
    {}
    """

    from pygments.lexers import get_lexer_for_filename
    lexer = get_lexer_for_filename(filename, stripall=True)

    from pygments.formatters import HtmlFormatter, TerminalFormatter
    from pygments import highlight

    try:
        from IPython.display import HTML
        formatter = HtmlFormatter(linenos=True, cssclass="source")
        html_code = highlight(code, lexer, formatter)
        css = formatter.get_style_defs()
        html = HTML_TEMPLATE.format(css, html_code)
        htmlres = HTML(html)

        return htmlres

    except Exception as e:
        print(e)
        pass

    formatter = TerminalFormatter()
    output = highlight(code,lexer,formatter)
    return output
