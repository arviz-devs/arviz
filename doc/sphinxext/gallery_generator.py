"""
Sphinx plugin to run example scripts and create a gallery page.

Modified from the seaborn project, which modified the mpld3 project.

Also inspired in bokeh's bokeh_gallery sphinxext.
"""
import glob
import os
import os.path as op
import re
import shutil
import token
import tokenize

import matplotlib
import matplotlib.pyplot as plt
from bokeh.io import export_png
from bokeh.layouts import gridplot
from matplotlib import image
from numpy import ndarray

from arviz.rcparams import rc_context

matplotlib.use("Agg")


MPL_RST_TEMPLATE = """
.. _{sphinx_tag}:

{docstring}

.. image:: {img_file}

**Python source code:** :download:`[download source: {fname}]<{fname}>`

**API documentation:** {api_name}

.. literalinclude:: {fname}
    :lines: {end_line}-
"""

BOKEH_RST_TEMPLATE = """
.. _{sphinx_tag}:

{docstring}

.. bokeh-plot:: {absfname}
    :source-position: none

**Python source code:** :download:`[download source: {fname}]<{fname}>`

**API documentation:** {api_name}

.. literalinclude:: {fname}
    :lines: {end_line}-
"""

RST_TEMPLATES = {"matplotlib": MPL_RST_TEMPLATE, "bokeh": BOKEH_RST_TEMPLATE}

BOKEH_EXPORT_CODE = """\n
if isinstance(ax, ndarray):
    if len(ax.shape) == 1:
        export_png(gridplot([ax.tolist()]), filename="{pngfilename}")
    else:
        export_png(gridplot(ax.tolist()), filename="{pngfilename}")
else:
    export_png(ax, filename="{pngfilename}")
"""

INDEX_TEMPLATE = """

.. raw:: html

    <style type="text/css">
    .figure {{
        position: relative;
        float: left;
        margin: 10px;
        width: 180px;
        height: 200px;
    }}

    .figure img {{
        position: absolute;
        display: inline;
        left: 0;
        width: 170px;
        height: 170px;
        opacity:1.0;
        filter:alpha(opacity=100); /* For IE8 and earlier */
    }}

    .figure:hover img {{
        -webkit-filter: blur(3px);
        -moz-filter: blur(3px);
        -o-filter: blur(3px);
        -ms-filter: blur(3px);
        filter: blur(3px);
        opacity:1.0;
        filter:alpha(opacity=100); /* For IE8 and earlier */
    }}

    .figure span {{
        position: absolute;
        display: inline;
        left: 0;
        width: 170px;
        height: 170px;
        background: #000;
        color: #fff;
        visibility: hidden;
        opacity: 0;
        z-index: 100;
    }}

    .figure p {{
        position: absolute;
        top: 45%;
        width: 170px;
        font-size: 110%;
    }}

    .figure:hover span {{
        visibility: visible;
        opacity: .4;
    }}

    .caption {{
        position: absolue;
        width: 180px;
        top: 170px;
        text-align: center !important;
    }}
    </style>

.. _{sphinx_tag}:

Example gallery
===============

{toctrees_contents}

"""


def create_thumbnail(infile, thumbfile, width=275, height=275, cx=0.5, cy=0.5, border=4):
    baseout, extout = op.splitext(thumbfile)

    im = image.imread(infile)
    rows, cols = im.shape[:2]
    size = min(rows, cols)
    if size == cols:
        xslice = slice(0, size)
        ymin = min(max(0, int(cx * rows - size // 2)), rows - size)
        yslice = slice(ymin, ymin + size)
    else:
        yslice = slice(0, size)
        xmin = min(max(0, int(cx * cols - size // 2)), cols - size)
        xslice = slice(xmin, xmin + size)
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax = fig.add_axes([0, 0, 1, 1], aspect="auto", frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect="auto", resample=True, interpolation="bilinear")
    fig.savefig(thumbfile, dpi=dpi)
    plt.close(fig)


def indent(s, N=4):
    """indent a string"""
    return s.replace("\n", "\n" + N * " ")


class ExampleGenerator:
    """Tools for generating an example page from a file"""

    def __init__(self, filename, target_dir, backend, thumb_dir):
        self.filename = filename
        self.target_dir = target_dir
        self.thumb_dir = thumb_dir
        self.backend = backend
        self.thumbloc = 0.5, 0.5
        self.extract_docstring()
        with open(filename, "r") as fid:
            self.filetext = fid.read()

        outfilename = op.join(target_dir, self.rstfilename)

        # Only actually run it if the output RST file doesn't
        # exist or it was modified less recently than the example
        if not op.exists(outfilename) or (op.getmtime(outfilename) < op.getmtime(filename)):

            self.exec_file()
        else:

            print("skipping {0}".format(self.filename))

    @property
    def dirname(self):
        return op.split(self.filename)[0]

    @property
    def fname(self):
        return op.split(self.filename)[1]

    @property
    def modulename(self):
        return op.splitext(self.fname)[0]

    @property
    def pyfilename(self):
        return self.modulename + ".py"

    @property
    def rstfilename(self):
        return self.modulename + ".rst"

    @property
    def htmlfilename(self):
        return self.modulename + ".html"

    @property
    def pngfilename(self):
        pngfile = self.modulename + ".png"
        return "_images/" + pngfile

    @property
    def thumbfilename(self):
        pngfile = self.modulename + "_thumb.png"
        return pngfile

    @property
    def apiname(self):
        with open(op.join(self.target_dir, self.pyfilename), "r") as file:
            regex = r"az\.(plot\_[a-z_]+)\("
            name = re.findall(regex, file.read())
        apitext = name[0] if name else ""
        return (
            ":obj:`~arviz.{apitext}`".format(apitext=apitext)
            if apitext
            else "No API Documentation available"
        )

    @property
    def sphinxtag(self):
        return self.modulename

    @property
    def pagetitle(self):
        return self.docstring.strip().split("\n")[0].strip()

    def extract_docstring(self):
        """Extract a module-level docstring"""
        lines = open(self.filename).readlines()
        start_row = 0
        if lines[0].startswith("#!"):
            lines.pop(0)
            start_row = 1

        docstring = ""
        first_par = ""
        line_iter = lines.__iter__()
        tokens = tokenize.generate_tokens(lambda: next(line_iter))
        for tok_type, tok_content, _, (erow, _), _ in tokens:
            tok_type = token.tok_name[tok_type]
            if tok_type in ("NEWLINE", "COMMENT", "NL", "INDENT", "DEDENT"):
                continue
            elif tok_type == "STRING":
                docstring = eval(tok_content)
                # If the docstring is formatted with several paragraphs,
                # extract the first one:
                paragraphs = "\n".join(line.rstrip() for line in docstring.split("\n")).split(
                    "\n\n"
                )
                if len(paragraphs) > 0:
                    first_par = paragraphs[0]
            break

        thumbloc = None
        for i, line in enumerate(docstring.split("\n")):
            m = re.match(r"^_thumb: (\.\d+),\s*(\.\d+)", line)
            if m:
                thumbloc = float(m.group(1)), float(m.group(2))
                break
        if thumbloc is not None:
            self.thumbloc = thumbloc
            docstring = "\n".join([l for l in docstring.split("\n") if not l.startswith("_thumb")])

        self.docstring = docstring
        self.short_desc = first_par
        self.end_line = erow + 1 + start_row

    def exec_file(self):
        print("running {0}".format(self.filename))

        thumbfile = op.join(self.thumb_dir, self.thumbfilename)
        cx, cy = self.thumbloc
        pngfile = op.join(self.target_dir, self.pngfilename)
        plt.close("all")
        if self.backend == "matplotlib":
            my_globals = {"pl": plt, "plt": plt}
            with open(self.filename, "r") as fp:
                code_text = fp.read()
                code_text = re.sub(r"(plt\.show\S+)", "", code_text)
                exec(compile(code_text, self.filename, "exec"), my_globals)

            fig = plt.gcf()
            fig.canvas.draw()
            fig.savefig(pngfile, dpi=75)

        elif self.backend == "bokeh":
            pngfile = thumbfile
            with open(self.filename, "r") as fp:
                code_text = fp.read()
                code_text += BOKEH_EXPORT_CODE.format(pngfilename=thumbfile)
                with rc_context(rc={"plot.bokeh.show": False}):
                    exec(
                        code_text,
                        {"export_png": export_png, "ndarray": ndarray, "gridplot": gridplot},
                    )

        create_thumbnail(pngfile, thumbfile, cx=cx, cy=cy)

    def toctree_entry(self):
        return "   ./%s\n\n" % op.join(self.backend, op.splitext(self.htmlfilename)[0])

    def contents_entry(self):
        return (
            ".. raw:: html\n\n"
            "    <div class='figure align-center'>\n"
            "    <a href=./{0}/{1}>\n"
            "    <img src=../_static/{2}>\n"
            "    <span class='figure-label'>\n"
            "    <p>{3}</p>\n"
            "    </span>\n"
            "    </a>\n"
            "    </div>\n\n"
            "\n\n"
            "".format(self.backend, self.htmlfilename, self.thumbfilename, self.sphinxtag)
        )


def main(app):
    working_dir = os.getcwd()
    os.chdir(app.builder.srcdir)
    static_dir = op.join(app.builder.srcdir, "..", "build", "_static")
    target_dir_orig = op.join(app.builder.srcdir, "examples")

    backends = ("matplotlib", "bokeh")
    backend_titles = ("Matplotlib", "Bokeh")
    toctrees_contents = ""
    for backend_i, backend in enumerate(backends):
        target_dir = op.join(target_dir_orig, backend)
        image_dir = op.join(target_dir, "_images")
        thumb_dir = op.join(app.builder.srcdir, "example_thumbs")
        source_dir = op.abspath(op.join(app.builder.srcdir, "..", "..", "examples", backend))
        if not op.exists(static_dir):
            os.makedirs(static_dir)

        if not op.exists(target_dir):
            os.makedirs(target_dir)

        if not op.exists(image_dir):
            os.makedirs(image_dir)

        if not op.exists(thumb_dir):
            os.makedirs(thumb_dir)

        if not op.exists(source_dir):
            os.makedirs(source_dir)

        title = backend_titles[backend_i]
        toctree = ("\n\n{title}\n{underline}\n" "\n\n" ".. toctree::\n" "   :hidden:\n\n").format(
            title=title, underline="-" * len(title)
        )
        contents = "\n\n"

        # Write individual example files
        files = sorted(glob.glob(op.join(source_dir, "*.py")))
        for filename in files:

            ex = ExampleGenerator(filename, target_dir, backend, thumb_dir)

            shutil.copyfile(filename, op.join(target_dir, ex.pyfilename))
            output = RST_TEMPLATES[backend].format(
                sphinx_tag=ex.sphinxtag,
                docstring=ex.docstring,
                end_line=ex.end_line,
                fname=ex.pyfilename,
                absfname=op.join(target_dir, ex.pyfilename),
                img_file=ex.pngfilename,
                api_name=ex.apiname,
            )
            with open(op.join(target_dir, ex.rstfilename), "w") as f:
                f.write(output)

            toctree += ex.toctree_entry()
            contents += ex.contents_entry()

        toctrees_contents += "\n".join((toctree, contents))
        toctrees_contents += """.. raw:: html\n\n    <div style="clear: both"></div>"""

    # write index file
    index_file = op.join(target_dir, "..", "index.rst")

    with open(index_file, "w") as index:
        index.write(
            INDEX_TEMPLATE.format(
                sphinx_tag="example_gallery",
                toctrees_contents=toctrees_contents,
                examples_source=source_dir,
            )
        )
    shutil.copytree(thumb_dir, static_dir, dirs_exist_ok=True)

    os.chdir(working_dir)


def setup(app):
    app.connect("builder-inited", main)
