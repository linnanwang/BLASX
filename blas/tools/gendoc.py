#! /usr/bin/env python
#
# Sample usage:
#
# python tools/gendoc src/z*.cpp
#
# the visit "htmldoc/index.html"
#

import sys
import os
import re

def is_all_char(s, ch):
  """
  All characters in 's' are 'ch'.
  """

  for c in s:
    if c != ch:
      return 0

  return 1

def cleanup(txt):
  return "\n".join(cleanup_lst(txt.split("\n")))

def cleanup_lst(txtlst):
  last_empty = 0
  only_empty = 1
  empty_line = 0
  i = 0
  first_full = len(txtlst)

  if len(txtlst) > 1:
    min_indent = max(map(len, txtlst))
  else:
    min_indent = 0

  for line in txtlst:
    if len(line.strip()) == 0: # empty line
      if not empty_line:
        last_empty = i
      empty_line = 1
    else:
      if only_empty:
        first_full = i
        only_empty = 0
      empty_line = 0

    if not empty_line:
      min_indent = min(min_indent,  len(line) - len(line.lstrip()))

    i += 1

  all_lines = list()

  i = 0
  for line in txtlst:
    if i >= first_full and i <= last_empty:
      all_lines.append(line.rstrip()[min_indent:])
    i += 1

  return all_lines

def parse_args(argstr):
  arglst = argstr.split(",")
  newarglist = list()
  for a in arglst:
    a = a.strip()
    name = re.match("(\w+)", a[::-1]).group() # match argument name from the end (I'm assuming the name ends each argument)
    prefix = a[:-len(name)]
    newarglist.append((prefix, name[::-1]))

  return newarglist

class FuncDoc:
  def __init__(self, funcname, C_args, argdict, purpose, details):
    self.funcname = funcname
    self.C_args = C_args
    self.argdict = argdict
    self.purpose = purpose
    self.details = details

def getdoc(fname):
  txt = open(fname).read()
  purpose_idx = txt.find("\n", txt.find("Purpose")+14)+1
  args_idx = txt.find("Arguments", purpose_idx)
  func_idx = txt.rfind('extern "C"', 1, purpose_idx)
  funcname = txt[txt.find("\n", func_idx)+1 : txt.find("(", func_idx)]
  C_args_idx = txt.find("(", func_idx) + 1
  # I'm counting on the fact that there is no ')' in arg list
  C_args = parse_args(txt[C_args_idx : txt.find(")", C_args_idx)])

  details_idx = txt.find("   Further Details", args_idx)

  eoc_idx = txt.find("*/", args_idx)
  argend_idx = eoc_idx
  if details_idx > 0:
    argend_idx = details_idx

  if details_idx > 0:
    details_idx = txt.find("\n", details_idx + 30) + 1

  if details_idx > 0:
    details = cleanup(txt[details_idx:eoc_idx])
  else:
    details = ""

  argdict = dict()
  argname = "_____" # dummy argument name
  argdict[argname] = list()

  # go through each "argline" in the section with arguments
  for argline in txt[args_idx:argend_idx].split("\n"):
    argfields = argline.split()

    switcharg = 0
    #for inout in ("(input)", "(output)", "(input/output)", "(workspace)", "(input/workspace)", "(workspace/output)", "(input"):
    for inout in ("input", "output", "workspace"):
      #if len(argfields) > 1 and inout == argfields[1]:
      if re.match(" *\w+ *\(%s" % inout, argline):
        argname = argfields[0]
        arginout = argline[argline.find("(") : argline.find(")")+1]
        argtype = argline[argline.find(")")+1 :].strip()
        argdict[argname] = [(arginout, argtype)]
        switcharg = 1
        break

    if not switcharg:
      argdict[argname].append(argline)

  for key in argdict:
    inout = argdict[key][0]
    argdict[key] = [inout] + cleanup_lst(argdict[key][1:])
    continue
    l = argdict[key]
    idx = len(l) - 1
    fields = l[idx].split()
    if len(fields) == 1 and is_all_char(fields[0], "="):
      del l[idx]

  purpose = cleanup(txt[purpose_idx:args_idx])

  return FuncDoc(funcname, C_args, argdict, purpose, details)

# this function is out of date
def getlatexdoc(funcdoc):
  latexdoc = "\\textsf{magma\_int\_t} "
  latexdoc += "\\textsf{\\textbf{%s}}" % funcdoc.funcname.replace("_", "\\_")
  latexdoc += "(\\textsf{"
  latexdoc += ", ".join(funcdoc.C_args).replace("_", "\\_")
  latexdoc += "}); \\\n"

  latexdoc += "Arguments:\\\n\\begin{description}"

  for arg in funcdoc.C_args:
    argname = ""
    larg = len(arg)
    for idx in range(larg):
      ch = arg[larg-idx-1]
      if ch.isalpha():
        argname = ch + argname
      else:
        break

    latexdoc += "\\item[" + argname + "] "

    karg = argname.upper()
    if 0 and not funcdoc.argdict.has_key(karg):
      karg = "D" + karg

    try:
      ldoc = funcdoc.argdict[karg]
      latexdoc += ldoc[0] + "\n" # input/output/workspace
      latexdoc += "\\begin{verbatim}\n"

      lidx = len(ldoc[1])
      # remove extra indentation
      for l in ldoc[1:]:
        if len(l) > 0:
          lidx = min(lidx, len(l) - len(l.lstrip()))

      for l in ldoc[1:]:
        if l.rstrip():
          latexdoc += l[lidx:] + "\n"
      latexdoc += "\end{verbatim}\n"

    except:
      #sys.stderr.write("MAGMA %s\n" % " ".join(list((funcdoc.funcname, karg, str(funcdoc.argdict.keys()), latexdoc, str(funcdoc.argdict)))))
      pass

    latexdoc += "\n"

  latexdoc += "\\end{description}"

  return latexdoc

class HtmlDoc:
  def tag(self, tag, s):
    return "<" + tag + ">" + s + "</" + tag.split()[0] + ">\n"

  def preamble(self, title):
    return """<html>
<head>
<title>%s</title>
<style>
b {
font-family: sans-serif;
color: red;
}
em {
font-family: sans-serif;
color: blue;
}
table {
white-space: nowrap;
font-weight: bold;
margin-left: 6px;
border-top: 1px solid #A8B8D9;
border-left: 1px solid #A8B8D9;
border-right: 1px solid #A8B8D9;
padding: 6px 0px 6px 0px;
color: #253555;
font-weight: bold;
text-shadow: 0px 1px 1px rgba(255, 255, 255, 0.9);
/* firefox specific markup */
-moz-box-shadow: rgba(0, 0, 0, 0.15) 5px 5px 5px;
-moz-border-radius-topright: 8px;
-moz-border-radius-topleft: 8px;
/* webkit specific markup */
-webkit-box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.15);
-webkit-border-top-right-radius: 8px;
-webkit-border-top-left-radius: 8px;
background-image:url('nav_f.png');
background-repeat:repeat-x;
background-color: #E2E8F2;
}
pre {background: #ffffee; border: 1px solid silver; padding: 0.5em;}
</style>
</head>
<body>
""" % title

  def epilog(self, title):
    return """</body>
</html>
"""

  def begin_decl(self):
    return "<table>\n"

  def end_decl(self):
    return "</table>\n"

  def ret_type(self, s):
    return "<tr><td>" + s + " "
    return self.tag('div class="ret_type"', s)

  def funcname(self, s):
    return self.tag('b', s) + "</td>"
    return self.tag('div class="funcname"', s)

  def begin_purpose(self, s):
    return "<p><h2>%s</h2>\n<pre>\n" % s

  def end_purpose(self, s):
    return "</pre>\n</p>\n"

  def begin_arguments(self, s):
    return "<p><h2>%s</h2>\n" % s
    return self.tag('div class="arguments"', s) + "<ul>\n"

  def end_arguments(self, s):
    return "</ul>\n"

  def begin_arg(self, argname):
    return "<li><b>" + argname + "</b>\n"

  def end_arg(self, argname):
    return ""

  def inoutwork(self, s):
    return s + "\n"

  def arg_type(self, s):
    return self.tag("em", s)

  def begin_arg_desc(self, argname):
    return "<pre>\n"

  def arg_desc_line(self, s):
    return s + "\n"

  def end_arg_desc(self, argname):
    return "</pre>\n</li>\n"

  def begin_details(self, s):
    return "<p><h2>%s</h2>\n<pre>\n" % s

  def end_details(self, s):
    return "</pre>\n</p>\n"

  def begin_args_decl(self, argcount):
    self.argcount = argcount
    return "<td>(</td>"

  def end_args_decl(self):
    return "<td>)</td>\n</tr>\n"

  def decl_arg(self, prefix, argname):
    return "<td>" + prefix + " " + self.tag("em", argname) + "<td>\n</tr>\n<tr><td></td><td></td>"

def getoutputdoc(fdoc, output):
  doc = output.begin_decl()
  doc += output.ret_type("magma_int_t")
  doc += output.funcname(fdoc.funcname)
  doc += output.begin_args_decl(len(fdoc.C_args))
  doc += "".join(map(lambda s: output.decl_arg(s[0], s[1]), fdoc.C_args))
  doc += output.end_args_decl()
  doc += output.end_decl()

  doc += output.begin_purpose("Purpose:")
  doc += fdoc.purpose
  doc += output.end_purpose("Purpose:")

  doc += output.begin_arguments("Arguments:")

  for arg in fdoc.C_args:
    argname = arg[1]
    doc += output.begin_arg(argname)
    doc += output.end_arg(argname)

    karg = argname.upper()
    if 0 and not funcdoc.argdict.has_key(karg):
      karg = "D" + karg

    ldoc = fdoc.argdict.get(karg, [("(missing)", arg[0]), "MISSING"])
    #sys.stderr.write("MAGMA %s\n" % " ".join(list((fdoc.funcname, karg, str(fdoc.argdict.keys()), doc, str(fdoc.argdict)))))

    if len(ldoc) > 1:
      lidx = len(ldoc[1])
    else:
      lidx = 0
    # remove extra indentation
    for l in ldoc[1:]:
      if len(l) > 0:
        lidx = min(lidx, len(l) - len(l.lstrip()))

    doc += output.inoutwork(ldoc[0][0]) # input/output/workspace
    doc += output.arg_type(ldoc[0][1])
    doc += output.begin_arg_desc(argname)

    for l in ldoc[1:]:
      if l.rstrip():
        doc += output.arg_desc_line(l[lidx:])

    doc += output.end_arg_desc(argname)


  doc += output.end_arguments("Arguments:")

  if fdoc.details:
    doc += output.begin_details("Further Details:")
    doc += fdoc.details
    doc += output.end_details("Further Details:")

  return doc

def main(argv):
  outdir = "htmldoc"

  if not os.path.exists(outdir):
    os.mkdir(outdir)

  fl = list()
  for fname in argv[1:]:
    ofname = os.path.basename(fname).replace("cpp", "html")
    of = open(os.path.join(outdir, ofname), "w")

    odoc = HtmlDoc()

    of.write(odoc.preamble("MAGMA"))

    funcdoc = getdoc(fname)
    #latexdoc = getlatexdoc(funcdoc)
    doc = getoutputdoc(funcdoc, odoc)
    of.write(doc)
    of.write(odoc.epilog("MAGMA"))

    fl.append((funcdoc.funcname, ofname))

  idxf = open(os.path.join(outdir, "index.html"), "w")
  idxf.write("<html>\n<head><title>MAGMA 1.0</title></head>\n<body>\n<h1>MAGMA 1.0 Function Index</h1><ul>\n")
  for t in fl:
    funcname, ofname = t
    idxf.write("""<li><a href="%s"><code>%s</code></a></li>\n""" % (ofname, funcname))
  idxf.write("</ul>\n</body>\n</html>\n")

  return 0

if "__main__" == __name__ :
  sys.exit(main(sys.argv))
