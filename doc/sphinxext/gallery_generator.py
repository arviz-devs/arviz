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

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import image

from arviz.rcparams import rc_context
from arviz import rcParams

matplotlib.use("Agg")


MPL_RST_TEMPLATE = """
.. _{sphinx_tag}:

{docstring}

**API documentation:** {api_name}

.. tabbed:: Matplotlib

    .. image:: {img_file}

    **Python source code:** :download:`[download source: {fname}]<{fname}>`

    .. literalinclude:: {fname}
        :lines: {end_line}-

"""

BOKEH_RST_TEMPLATE = """
.. tabbed:: Bokeh

    .. bokeh-plot:: {absfname}
        :source-position: none

    **Python source code:** :download:`[download source: {fname}]<{fname}>`

    .. literalinclude:: {fname}
        :lines: {end_line}-
"""

RST_TEMPLATES = {"matplotlib": MPL_RST_TEMPLATE, "bokeh": BOKEH_RST_TEMPLATE}

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

    span.figure-label {{
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
        position: absolute;
        width: 180px;
        top: 170px;
        text-align: center !important;
    }}

    .figure .gallery-figure-title p {{
        position: relative;
        top: 170px;
        color: black;
        visibility: visible;
        text-align: center !important;
        line-height: normal;
    }}
    .figure .gallery-figure-title span {{
        top: 170px;
        position: relative;
        visibility: visible;
    }}
    </style>

.. _{sphinx_tag}:

Example gallery
===============

{toctrees_contents}

"""

CONTENTS_ENTRY_TEMPLATE = (
    ".. raw:: html\n\n"
    "    <div class='figure align-center'>\n"
    "    <a href=./{htmlfilename}>\n"
    "    <img src=../_static/{thumbfilename}>\n"
    "    <span class='figure-label'>\n"
    "    <p>{sphinx_tag}</p>\n"
    "    </span>\n"
    '    <span class="gallery-figure-title">\n'
    "      <p>{title}</p>\n"
    "    </span>\n"
    "    </a>\n"
    "    </div>\n\n"
    "\n\n"
    ""
)


def create_thumbnail(infile, thumbfile, width=275, height=275, cx=0.5, cy=0.5, border=4):

    im = image.imread(infile)
    rows, cols = im.shape[:2]
    size = min(rows, cols)
    if size == cols:
        xslice = slice(0, size)
        ymin = min(max(0, int(cy * rows - size // 2)), rows - size)
        yslice = slice(ymin, ymin + size)
    else:
        yslice = slice(0, size)
        xmin = min(max(0, int(cx * cols - size // 2)), cols - size)
        xslice = slice(xmin, xmin + size)
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, constrained_layout=True)

    ax = fig.add_axes([0, 0, 1, 1], aspect="auto", frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect="auto", resample=True, interpolation="bilinear")
    fig.savefig(thumbfile, dpi=dpi)
    plt.close(fig)


def indent(s, N=4):
    """indent a string"""
    return s.replace("\n", "\n" + N * " ")


class ExampleGenerator:
    """Tools for generating an example page from a file"""

    _title: Optional[str]

    def __init__(self, filename, target_dir, backend, thumb_dir, target_dir_orig):
        self.filename = filename
        self.target_dir = target_dir
        self.thumb_dir = thumb_dir
        self.backend = backend
        self.thumbloc = 0.5, 0.5
        self._title = None
        self.extract_docstring()
        with open(filename, "r") as fid:
            self.filetext = fid.read()

        outfilename = op.join(target_dir_orig, self.rstfilename)

        # Only actually run it if the output RST file doesn't
        # exist or it was modified less recently than the example
        if not op.exists(outfilename) or (op.getmtime(outfilename) < op.getmtime(filename)):

            self.exec_file()
        else:

            print("skipping {0}".format(self.filename))

    @property
    def title(self) -> str:
        if self._title is not None:
            return self._title
        return self.modulename

    @property
    def dirname(self):
        return op.split(self.filename)[0]

    @property
    def fname(self):
        return op.split(self.filename)[1]

    @property
    def modulename(self) -> str:
        return op.splitext(self.fname)[0]

    @property
    def basename(self) -> str:
        return self.modulename.split("_", 1)[1]

    @property
    def pyfilename(self):
        return self.modulename + ".py"

    @property
    def rstfilename(self):
        return self.basename + ".rst"

    @property
    def htmlfilename(self):
        return self.basename + ".html"

    @property
    def pngfilename(self):
        pngfile = self.modulename + ".png"
        return "_images/" + pngfile

    @property
    def thumbfilename(self):
        pngfile = self.basename + "_thumb.png"
        return pngfile

    @property
    def apiname(self):
        with open(op.join(self.target_dir, self.pyfilename), "r") as file:
            regex = r"az\.(plot\_[a-z_]+)\("
            name = re.findall(regex, file.read())
        apitext = name[0] if name else ""
        return (
            ":func:`~arviz.{apitext}`".format(apitext=apitext)
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
        title: Optional[str] = None
        ex_title: str = ""
        for line in docstring.split("\n"):
            # we've found everything we need...
            if thumbloc and title and ex_title != "":
                break
            m = re.match(r"^_thumb: (\.\d+),\s*(\.\d+)", line)
            if m:
                thumbloc = float(m.group(1)), float(m.group(2))
                continue
            m = re.match(r"^_example_title: (.*)$", line)
            if m:
                title = m.group(1)
                continue
            # capture the first non-empty line of the docstring as title
            if ex_title == "":
                ex_title = line
        assert ex_title != ""
        if thumbloc is not None:
            self.thumbloc = thumbloc
            docstring = "\n".join([l for l in docstring.split("\n") if not l.startswith("_thumb")])

        if title is not None:
            docstring = "\n".join(
                [l for l in docstring.split("\n") if not l.startswith("_example_title")]
            )
        else:
            title = ex_title
        self._title = title

        self.docstring = docstring
        self.short_desc = first_par
        self.end_line = erow + 1 + start_row  # pylint: disable=undefined-loop-variable

    def exec_file(self):
        # pylint: disable=exec-used
        print("running {0}".format(self.filename))

        plt.close("all")
        if self.backend == "matplotlib":
            thumbfile = op.join(self.thumb_dir, self.thumbfilename)
            cx, cy = self.thumbloc
            pngfile = op.join(self.target_dir, self.pngfilename)
            my_globals = {"plt": plt}
            with open(self.filename, "r") as fp:
                code_text = fp.read()
                code_text = re.sub(r"(plt\.show\S+)", "", code_text)
                exec(compile(code_text, self.filename, "exec"), my_globals)

            fig = plt.gcf()
            fig.canvas.draw()
            fig.savefig(pngfile, dpi=75)
            create_thumbnail(pngfile, thumbfile, cx=cx, cy=cy)

        elif self.backend == "bokeh":
            with open(self.filename, "r") as fp:
                code_text = fp.read()
                with rc_context(rc={"plot.bokeh.show": False}):
                    exec(code_text)

    def toctree_entry(self):
        return "   ./%s\n\n" % op.join(op.splitext(self.htmlfilename)[0])

    def contents_entry(self) -> str:
        return CONTENTS_ENTRY_TEMPLATE.format(
            backend=self.backend,
            htmlfilename=self.htmlfilename,
            thumbfilename=self.thumbfilename,
            sphinx_tag=self.sphinxtag,
            title=self.title,
        )


def main(app):
    working_dir = os.getcwd()
    os.chdir(app.builder.srcdir)
    static_dir = op.join(app.builder.srcdir, "..", "build", "_static")
    target_dir_orig = op.join(app.builder.srcdir, "examples")

    backends = ("matplotlib", "bokeh")
    backend_prefixes = ("mpl", "bokeh")
    toctrees_contents = ""
    thumb_dir = op.join(app.builder.srcdir, "example_thumbs")

    if not op.exists(static_dir):
        os.makedirs(static_dir)

    if not op.exists(thumb_dir):
        os.makedirs(thumb_dir)

    path_dict = {}
    for backend in backends:
        target_dir = op.join(target_dir_orig, backend)
        image_dir = op.join(target_dir, "_images")
        source_dir = op.abspath(op.join(app.builder.srcdir, "..", "..", "examples", backend))

        if not op.exists(source_dir):
            os.makedirs(source_dir)

        if not op.exists(target_dir):
            os.makedirs(target_dir)

        if not op.exists(image_dir):
            os.makedirs(image_dir)

        path_dict[backend] = {
            "source_dir": source_dir,
            "target_dir": target_dir,
            "image_dir": image_dir,
        }

    toctrees_contents = ""
    toctree = "\n\n.. toctree::\n   :hidden:\n\n"
    contents = "\n\n"

    # Write individual example files
    files = sorted(glob.glob(op.join(path_dict["matplotlib"]["source_dir"], "*.py")))
    for filename in files:
        base_filename = op.split(filename)[1].split("_", 1)[1]
        example_contents = ""
        for backend, prefix in zip(backends, backend_prefixes):
            source_dir = path_dict[backend]["source_dir"]
            target_dir = path_dict[backend]["target_dir"]
            expected_filename = op.join(source_dir, f"{prefix}_{base_filename}")

            if not op.exists(expected_filename):
                if backend == "matplotlib":
                    raise ValueError("All examples must have a matplotlib counterpart.")
                continue

            ex = ExampleGenerator(
                expected_filename, target_dir, backend, thumb_dir, target_dir_orig
            )

            shutil.copyfile(expected_filename, op.join(target_dir, ex.pyfilename))
            output = RST_TEMPLATES[backend].format(
                sphinx_tag=ex.sphinxtag,
                docstring=ex.docstring,
                end_line=ex.end_line,
                fname=op.join(backend, ex.pyfilename),
                absfname=op.join(target_dir, ex.pyfilename),
                img_file=op.join(backend, ex.pngfilename),
                api_name=ex.apiname,
            )
            example_contents += output

        with open(op.join(target_dir_orig, ex.rstfilename), "w") as f:
            f.write(example_contents)

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

    os.chdir(working_dir)


def setup(app):
    app.connect("builder-inited", main)
