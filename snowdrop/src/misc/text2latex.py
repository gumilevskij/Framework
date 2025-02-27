"""
    Simple Text to Latex converter.
    
    .. note::
        Please see: 
            https://github.com/rivermont/txt-to-tex
            https://stackoverflow.com/questions/8085520/generating-pdf-latex-with-python-script
"""
import os, sys
#from  misc.latex import eq2tex, greekify
from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, bold, NoEscape
#from pytexit import py2tex

PLATFORM = sys.platform

                
def getDocElements(model):
    """
    Get document elements.
    
    Args:
        model : Model object.
    """
    var_names    = model.symbols["variables"]
    var_values   = model.calibration["variables"]
    param_names  = model.symbols["parameters"]
    param_values = model.calibration["parameters"]
    shock_names  = model.symbols["shocks"]
    meas_shocks  = model.symbols.get("measurement_shocks",[])
    meas_vars    = model.symbols.get("measurement_variables",[])
    var_labels   = model.symbols.get("variables_labels",{})
    # shock_values = model.calibration["shocks"]
    eqs          = model.symbolic.equations
    constraints  = model.symbolic.constraints
    meas_eqs     = model.symbolic.measurement_equations
    if meas_eqs is None:
        meas_eqs = []
    eqs_comments   = model.symbols.get("equations_comments",[])
    
    var = []
    for n,v in zip(var_names,var_values):
        if "_plus_" in n or "_minus_" in n:
            continue
        x = "{} = {:.1f}".format(n,v)
        var.append(x)
    var = sorted(var)
    str_var = ", ".join(var)
    
    var = []
    if not model.steady_state is None:
        for n,v in zip(var_names,model.steady_state):
            x = "{} = {:.1f}".format(n,v)
            var.append(x)
        var = sorted(var)
    str_ss = ", ".join(var)
    
    
    labels = []
    keys = sorted(var_labels.keys())
    for k in keys:
       x = "{:<10}  --  {}".format(k,var_labels[k])
       labels.append(x) 
    labels = sorted(labels)
    str_labels = "\n".join(labels)
    
    par = []
    for n,v in zip(param_names,param_values):
        if 0<abs(v)<0.01 or abs(v) >= 100:
            x = "{} = {:.2e}".format(n,v)
        else:
            x = "{} = {:.2f}".format(n,v)
        par.append(x)
    par = sorted(par)
    str_par = ", ".join(par)
    
    # shocks = []
    # for n,v in zip(shock_names,shock_values):
    #     x = "{} = {:.1f}".format(n,v)
    #     shocks.append(x)
    # str_shocks = ", ".join(shocks)
    shock_names = sorted(shock_names)
    str_shocks = ", ".join(shock_names)
    
    meas_vars = sorted(meas_vars)
    str_meas_var = ", ".join(meas_vars)
    str_meas_shocks = ", ".join(meas_shocks)
    
    paragraphs = ['Endogenous Variables Values','Endogenous Variables Steady State Values','Measurement Variables','Parameters','Shocks','Measurement Shocks','Equations','Constraints', 'Measurement Equations','Legend']
    
    equations = []
    for i,eq in enumerate(eqs):
        if "_plus_" in eq or "_minus_" in eq:
            continue
        if i < len(eqs_comments) and not eqs_comments[i].isdigit():
            equations.append(eqs_comments[i])
        label = str(i+1)
        eq = eq.replace("**","^")
        e = u" {eqn:4} :  {eqs}\n".format(eqn=label, eqs=eq)
        #e = eq2tex(var_names,eq)
        equations.append(e)
    str_eqs = "\n".join(equations)
    
    cnstr = []
    if not constraints is None:
        for c in constraints:
            c = c.replace(".lt.", " < ")
            c = c.replace(".le.", " <= ")
            c = c.replace(".gt.", " > ")
            c = c.replace(".ge.", " >= ")
            cnstr.append(c)
    str_constraints = "\n".join(cnstr)
    
    meas_equations = []
    for i,eq in enumerate(meas_eqs):
        label = str(i+1)
        eq = eq.replace("**","^")
        e = u" {eqn:4} :  {eqs}\n".format(eqn=label, eqs=eq)
        #e = eq2tex(var_names,eq)
        meas_equations.append(e)
    str_meas_eqs = "\n".join(meas_equations)
    
    return paragraphs, str_var, str_ss, str_labels, str_par, str_shocks, str_eqs, str_constraints, str_meas_var, str_meas_shocks, str_meas_eqs


def getLatexDocument(model):
    """
    Convert text to latex format.
    
    Args:
        model : Model object.
        
    """
    paragraphs, str_var, str_ss, str_labels, str_par, str_shocks, str_eqs, str_constraints, str_meas_var, str_meas_shocks, str_meas_eqs = getDocElements(model)

    tex = [r"""\documentclass[12pt, letterpaper]{article}
    
    %UTF-8
    %\usepackage{utf8}
    
    %Margin - 1 inch on all sides
    \usepackage[letterpaper]{geometry}
    \usepackage{times}
    \geometry{top=1.0in, bottom=1.0in, left=1.0in, right=1.0in}
    
    %Doublespacing
    \usepackage{setspace}
    \doublespacing
    
    %Rotating tables (e.g. sideways when too long)
    \usepackage{rotating}
    
    %Fancy-header package to modify header/page numbering (insert last name)
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \lhead{}
    \chead{}
    \rhead{}
    \lfoot{}
    \cfoot{}
    \rfoot{}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
    %To make sure we actually have header 0.5in away from top edge
    %12pt is one-sixth of an inch. Subtract this from 0.5in to get headsep value
    \setlength\headsep{0.333in}
    
    %Environment
    %(to start, use \begin{workscited...}, each entry preceded by \bibent)
    \newcommand{\bibent}{\noindent \hangindent 40pt}
    \newenvironment{workscited}{\newpage \begin{center} Works Cited \end{center}}{\newpage }
    
    
    \begin{document}
    \begin{flushleft}
    
    %Title
    \begin{center}
    """, model.name, r"""
    \end{center}
    
    %Changes paragraph indentation to 0.5in
    \setlength{\parindent}{0.5in}
    
    %Begin body of document here
    
    """, paragraphs[0], r"""
    """, str_var, r"""
    
    """, paragraphs[1], r"""
    """, str_par, r"""
    
    """, paragraphs[2], r"""
    """, str_shocks, r"""
    
    """, paragraphs[3], r"""
    """, str_eqs, r"""
    
    \newpage
    
    \end{flushleft}
    \end{document}"""]
    
    return tex


def fill_document(doc,name,fname,paragraphs,elements):
    """Add a section, a subsection and some text to the document.

    :param doc: the document
    :type doc: :class:`pylatex.document.Document` instance
    
    """
    with doc.create(Section('Model Information')):
        doc.append('name: ')
        doc.append(italic(name))
        doc.append('\nfile: ')
        doc.append(italic(fname))
        
        for p,e in zip(paragraphs,elements):
            if bool(e):
                with doc.create(Subsection(p)):
                    doc.append(e)


def saveDocument(model):
    """Save model document."""
    name = model.infos['name']
    fname = model.infos['filename']
            
    paragraphs, str_var, str_ss, str_labels, str_par, str_shocks, str_eqs, str_constraints, str_meas_var, str_meas_shocks, str_meas_eqs = getDocElements(model)
    
    elements = [str_var, str_ss, str_meas_var, str_par, str_shocks, str_meas_shocks, str_eqs, str_constraints, str_meas_eqs, str_labels]
    
    doc = Document()
    
    doc.preamble.append(Command('title', bold('Model File')))
    doc.preamble.append(Command('author', italic('Generated by Python Framework')))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))
    
    fill_document(doc,name,fname,paragraphs,elements)

    name, ext = os.path.splitext(fname)
    model_path = fname.replace(name+ext,name)
    
    if PLATFORM == 'linux':
        doc.generate_pdf(filepath=model_path,clean_tex=True)
    else:
        doc.generate_tex(filepath=model_path)
   