## storm-analyzer ##
This is an attempt at a nice-looking, interactive GUI companion for the [Zhuang lab's](http://zhuang.harvard.edu/) 
[storm-analysis](https://github.com/ZhuangLab/storm-analysis/) project, built using the [Kivy](https://github.com/kivy/kivy)
toolkit. It is meant to allow interactive editing/visualization of STORM fitting parameters prior to batch analysis.
In its pre-packaged forms (currently only for macOS), it can act as a stand-alone tool to perform STORM analysis
(i.e. it packages the entire storm-analysis toolkit and does not require users to do any package installation or 
use the commandline).

## Features ##
- Drag-and-drop loading of images, STORM parameters, or molecule list files.
- Image viewer supports zoom, pan, and dragging to view the results of your fitting.
- Parameter fields taking a filename (such as CMOS calibration files) can be double-clicked to bring up a file chooser dialog.
- Full STORM analysis, including batch processing of directories, can be launched from the GUI.
- Editable parameters are pulled from your storm-analysis installation, so as storm-analysis parameters evolve, so does the editor.
- Edited parameters can be exported for use in other workflows.




