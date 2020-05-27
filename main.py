from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.slider import Slider
from kivy.uix.scrollview import ScrollView
from kivy.uix.scatter import Scatter
from kivy.uix.image import Image
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.tabbedpanel import TabbedPanelItem
from kivy.uix.treeview import TreeView, TreeViewNode, TreeViewLabel
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from kivy.properties import StringProperty, ObjectProperty
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.graphics import Line, Color, InstructionGroup
from kivy.graphics.transformation import Matrix
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.clock import Clock, mainthread
from kivy.uix.popup import Popup

# storm-analysis stuff
import storm_analysis.sa_library.datareader as datareader
import storm_analysis.sa_library.i3dtype as i3dtype
import storm_analysis.sa_library.readinsight3 as readinsight3
import storm_analysis.sa_library.sa_h5py as saH5Py
import storm_analysis.sa_library.parameters as params
import storm_analysis.daostorm_3d.find_peaks as find_peaks
import storm_analysis.sa_library.analysis_io as analysisIO

# For analysis
import storm_analysis.daostorm_3d.mufit_analysis as mfit
import storm_analysis.sa_utilities.batch_analysis as batch_analysis

# Third-party
from slider import RangeSlider
from progressSpinner import ProgressSpinner

# basic python
import numpy
import os
import sys
import copy
from functools import partial
import threading
import re


class Zoom(ScatterLayout):
    move_lock = False
    scale_lock_left = False
    scale_lock_right = False
    scale_lock_top = False
    scale_lock_bottom = False

    def on_touch_up(self, touch):

        self.move_lock = False
        self.scale_lock_left = False
        self.scale_lock_right = False
        self.scale_lock_top = False
        self.scale_lock_bottom = False
        if touch.grab_current is self:
            touch.ungrab(self)
            x = self.pos[0] / 10
            x = round(x, 0)
            x = x * 10
            y = self.pos[1] / 10
            y = round(y, 0)
            y = y * 10
            self.pos = x, y
            return super(Zoom, self).on_touch_up(touch)

    def transform_with_touch(self, touch):
        changed = False
        x = self.bbox[0][0]
        y = self.bbox[0][1]
        width = self.bbox[1][0]
        height = self.bbox[1][1]
        mid_x = x + width / 2
        mid_y = y + height / 2
        inner_width = width * 0.5
        inner_height = height * 0.5
        left = mid_x - (inner_width / 2)
        right = mid_x + (inner_width / 2)
        top = mid_y + (inner_height / 2)
        bottom = mid_y - (inner_height / 2)

            # just do a simple one finger drag
        if len(self._touches) == self.translation_touches:
            # _last_touch_pos has last pos in correct parent space,
            # just like incoming touch
            dx = (touch.x - self._last_touch_pos[touch][0]) \
                 * self.do_translation_x
            dy = (touch.y - self._last_touch_pos[touch][1]) \
                 * self.do_translation_y
            dx = dx / self.translation_touches
            dy = dy / self.translation_touches
            if (touch.x > left and touch.x < right and touch.y < top and touch.y > bottom or self.move_lock) and not self.scale_lock_left and not self.scale_lock_right and not self.scale_lock_top and not self.scale_lock_bottom:
                self.move_lock = True
                self.apply_transform(Matrix().translate(dx, dy, 0))
                changed = True

        change_x = touch.x - self.prev_x
        change_y = touch.y - self.prev_y
        anchor_sign = 1
        sign = 1
        if abs(change_x) >= 9 and not self.move_lock and not self.scale_lock_top and not self.scale_lock_bottom:
            if change_x < 0:
                sign = -1
            if (touch.x < left or self.scale_lock_left) and not self.scale_lock_right:
                self.scale_lock_left = True
                self.pos = (self.pos[0] + (sign * 10), self.pos[1])
                anchor_sign = -1
            elif (touch.x > right or self.scale_lock_right) and not self.scale_lock_left:
                self.scale_lock_right = True
            self.size[0] = self.size[0] + (sign * anchor_sign * 10)
            self.prev_x = touch.x
            changed = True
        if abs(change_y) >= 9 and not self.move_lock and not self.scale_lock_left and not self.scale_lock_right:
            if change_y < 0:
                sign = -1
            if (touch.y > top or self.scale_lock_top) and not self.scale_lock_bottom:
                self.scale_lock_top = True
            elif (touch.y < bottom or self.scale_lock_bottom) and not self.scale_lock_top:
                self.scale_lock_bottom = True
                self.pos = (self.pos[0], self.pos[1] + (sign * 10))
                anchor_sign = -1
            self.size[1] = self.size[1] + (sign * anchor_sign * 10)
            self.prev_y = touch.y
            changed = True

        return changed

    def on_touch_down(self, touch):
        x, y = touch.x, touch.y
        self.prev_x = touch.x
        self.prev_y = touch.y

       # if the touch isnt on the widget we do nothing
        if not self.do_collide_after_children:
            if not self.collide_point(x, y):
                return False

        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                ## zoom in
                if self.scale < 10:
                    self.scale = self.scale * 1.1

            elif touch.button == 'scrollup':
                if self.scale > 0.5:
                    self.scale = self.scale * 0.8


        # let the child widgets handle the event if they want
        touch.push()
        touch.apply_transform_2d(self.to_local)
        if super(Scatter, self).on_touch_down(touch):
            # ensure children don't have to do it themselves
            if 'multitouch_sim' in touch.profile:
                touch.multitouch_sim = True
            touch.pop()
            self._bring_to_front(touch)
            return True
        touch.pop()

        # if our child didn't do anything, and if we don't have any active
        # interaction control, then don't accept the touch.
        if not self.do_translation_x and \
                not self.do_translation_y and \
                not self.do_rotation and \
                not self.do_scale:
            return False

        if self.do_collide_after_children:
            if not self.collide_point(x, y):
                return False

        if 'multitouch_sim' in touch.profile:
            touch.multitouch_sim = True

        # grab the touch so we get all it later move events for sure
        self._bring_to_front(touch)
        touch.grab(self)
        self._touches.append(touch)
        self._last_touch_pos[touch] = touch.pos
        return True


class DaxViewer(Zoom):
    def __init__(self, **kwds):
        super(DaxViewer, self).__init__(**kwds)
        self.data = False
        self.image = False
        self.movie_file = None
        self.cur_frame = 0
        self.directory = ""
        self.film_l = 0
        self.film_x = 255
        self.film_y = 255
        self.locs1_list = None
        self.locs2_list = None
        self.locs = []
        self.drawing = None
        self.nm_per_pixel = 167
        self.analyze_thread = None
        self.progressWidget = None
        self.daopath = ""
        self._popup = None

    def addDax(self, filepath):
        self.directory = os.path.dirname(filepath)
        self.movie_file = datareader.inferReader(filepath)

        self.directory = os.path.dirname(filepath)
        self.movie_file = datareader.inferReader(filepath)
        [self.film_x, self.film_y, self.film_l] = self.movie_file.filmSize()
        self.slider_curframe.max = self.film_l - 1
        self.cur_frame = 0

        # Clear molecule lists.
        for elt in [self.locs1_list, self.locs2_list]:
            if elt is not None:
                elt.cleanUp()

        frame = self.movie_file.loadAFrame(0)
        self.slider_contrast.min = float(frame.min())
        self.slider_contrast.max = float(frame.max())
        self.slider_contrast.value1 = float(frame.min())
        self.slider_contrast.value2 = float(frame.max())
        self.slider_curframe.value = 0

        self.locs1_list = None
        self.locs2_list = None

        self.incCurFrame(0)


    def newFrame(self, frame, locs1, locs2, fmin, fmax):
        ## process image
        # save image
        self.data = frame.copy()

        # scale image.
        frame = 255.0*(frame-fmin)/(fmax-fmin)
        frame[(frame > 255.0)] = 255.0
        frame[(frame < 0.0)] = 0.0

        # Get image into Widget
        frame = numpy.ascontiguousarray(frame.astype(numpy.uint8))
        h, w = frame.shape
        frame_RGB = numpy.zeros((frame.shape[0], frame.shape[1], 3), dtype = numpy.uint8)
        frame_RGB[:,:,0] = frame
        frame_RGB[:,:,1] = frame
        frame_RGB[:,:,2] = frame

        texture = Texture.create(size=(w,h), colorfmt="rgb")
        texture.blit_buffer(frame_RGB.tostring(), bufferfmt="ubyte", colorfmt="rgb")

        if self.image:
            self.image.texture = texture
        else:
            self.image = Image(texture=texture)
            self.add_widget(self.image)
            self.bind(size=self.redrawFits)

        # Display localizations

        # Made it to locs
        for loc in locs1:
            self.canvas.add(loc)
            #self.canvas.add(loc)


        # Work this out to display in Kivy
        #self.image = QtGui.QImage(frame_RGB.data, w, h, QtGui.QImage.Format_RGB32)
        #self.image.ndarray1 = frame
        #self.image.ndarray2 = frame_RGB

    def displayFrame(self, update_locs):
        if self.movie_file:

            # Get the current frame.
            frame = self.movie_file.loadAFrame(self.cur_frame).astype(numpy.float)

            # Create localization list 1 molecule items.
            try:
                self.nm_per_pixel = float(self.parameters.getAttr("pixel_size"))
            except:
                self.nm_per_pixel=167

            # Clear drawings
            self.cleanDrawing()
            self.locs = []

            if update_locs and (self.locs1_list is not None):
                self.locs = self.locs1_list.createMolItems(self.cur_frame, self.nm_per_pixel)

            # Create localization list 2 molecule items.
            locs2 = []
            #if update_locs and (self.locs2_list is not None):
            #    locs2 = self.locs2_list.createMolItems(self.cur_frame, nm_per_pixel, self.image.pos)

            # Create/update Image widget
            #self.locs = locs1

            self.newFrame(frame,
                            self.locs,
                            locs2,
                            self.slider_contrast.value1,
                            self.slider_contrast.value2)


    def redrawFits(self, width, height):
        if self.locs1_list:
            self.cleanDrawing()
            self.locs = self.locs1_list.createMolItems(self.cur_frame, self.nm_per_pixel)

            for loc in self.locs:
                self.canvas.add(loc)

    def cleanDrawing(self):
        for i in self.locs:
            self.canvas.remove(i)

    def incCurFrame(self, amount):
        self.cur_frame += amount
        if (self.cur_frame < 0):
            self.cur_frame = 0
        if (self.cur_frame >= self.film_l):
            self.cur_frame = self.film_l - 1
        if self.movie_file:
            #self.ui.frameLabel.setText("frame " + str(self.cur_frame+1) + " (" + str(self.film_l) + ")")
            self.displayFrame(True)
            #self.locs_display_timer.start()

    def analyzeFrame(self):

        try:
            lparams=copy.deepcopy(self.parameters)
            lparams.setAttr('start_frame', 'int', self.cur_frame)
            lparams.setAttr('max_frame', 'int', self.cur_frame+2)

            finder = find_peaks.initFindAndFit(lparams)
            movie_reader = analysisIO.MovieReader(frame_reader = analysisIO.FrameReaderStd(movie_file=self.movie_file.filmFilename(), \
                                                                                            parameters=lparams), parameters = lparams)

            movie_reader.setup(self.cur_frame-1)
            movie_reader.nextFrame()
            peaks = finder.analyzeImage(movie_reader)

            self.locs1_list = MoleculeListSingle(peaks, self.cur_frame, self)
            self.redrawFits(None, None)
            #self.displayFrame(True)
            #self.incCurFrame(0) # Assuming this is to update display?

        except:
            return


    @mainthread
    def getDaxResults(self, hdfname):
        # Load in localizations
        self.locs1_list = None

        self.directory = os.path.dirname(hdfname)
        if saH5Py.isSAHDF5(hdfname):
            self.locs1_list = MoleculeListHDF5(hdfname,self)
        else:
            self.locs1_list = MoleculeListI3(hdfname,self)
        #self.locs1_table.showFields(self.locs1_list.getFields())
        #self.displayFrame(True)
        self.redrawFits(None, None)


    def analyzeDax(self):
        try:
            hdfname = os.path.basename(self.movie_file.filmFilename()[:-4] + ".h5")
            list_filename = self.directory + "/" + hdfname

            if os.path.exists(list_filename):
                os.remove(list_filename)

            self.parameters.toXMLFile(self.directory + "/" + 'current.xml')

            # Pass along analysis to a thread
            self.analyze_thread = threading.Thread(target = self.daoBackground,
                                              args=(self.movie_file.filmFilename(), list_filename,
                                                    self.directory + "/" + "current.xml"),
                                                    daemon=False)

            # Progress widget
            self.progressWidget = FloatLayout()
            progressWidget = ProgressSpinner(size_hint=(.05,.05), pos_hint = {'x': .94, 'y': .85})
            self.progressWidget.add_widget(progressWidget)
            self.parent.add_widget(self.progressWidget)
            self.analyze_thread.start()

        except:
            return


    def daoBackground(self, moviefile, hdfname, xmlfile):
        try:
            mfit.analyze(moviefile, hdfname, xmlfile)
            self.getDaxResults(hdfname)
            self.parent.remove_widget(self.progressWidget)
        except:
            self.parent.remove_widget(self.progressWidget)
            return



    def daostormBackground(self, analyzedir):
        batch_analysis.batchAnalysis(self.daopath + "/mufit_analysis.py", analyzedir + "/", analyzedir + "/", analyzedir + "/" + "current.xml",
                                     max_processes=8)

        self.parent.remove_widget(self.progressWidget)


        content = PopupDialog(ok=self.dismiss_popup)
        content.messagelabel.text = "Finished batch analysis"
        self._popup = Popup(title="Batch analysis", content=content,
                                size_hint=(.3, .3), pos_hint= {"center_x":0.5,"center_y":.5})

        self._popup.open()


    def dismiss_popup(self):
        self._popup.dismiss()

    def batchDaoBackground(self, analyzedir):
        try:
            self.parameters.toXMLFile(analyzedir + "/" + 'current.xml')
            self.daostorm_thread = threading.Thread(target = self.daostormBackground, args=(analyzedir,))

            self.progressWidget = FloatLayout()
            progressWidget = ProgressSpinner(size_hint=(.05,.05), pos_hint = {'x': .94, 'y': .85})
            self.progressWidget.add_widget(progressWidget)
            self.parent.add_widget(self.progressWidget)

            self.daostorm_thread.start()

        except:
            return


class Interface(BoxLayout):
    daxviewer = ObjectProperty(None)

    emccd_params = ObjectProperty(None)
    analysis_panel = ObjectProperty(None)
    panels = ["ParametersDAO",
              "ParametersSCMOS",
              "ParametersL1H",
              "ParametersMultiplane",
              "ParametersMultiplaneArb",
              "ParametersMultiplaneDao",
              "ParametersSpliner",
              "ParametersSplinerFISTA",
              "ParametersPSFFFT",
              "ParametersPupilFn"]

    # Mapping between Z-fit params and storm-analysis parameters
    zfitvars = {
            'wx0': 'wx_wo',
            'gx': 'wx_c',
            'zrx': 'wx_d',
            'Ax': 'wxA',
            'Bx': 'wxB',
            'Cx': 'wxC',
            'Dx': 'wxD',
            'wy0': 'wy_wo',
            'gy': 'wy_c',
            'zry': 'wy_d',
            'Ay': 'wyA',
            'By': 'wyB',
            'Cy': 'wyC',
            'Dy': 'wyD',
        }

    checkboxes = ["no_fitting", "drift_correction", "do_zfit", "z_correction"]
    manualwidgets = ["model", checkboxes]

    # Assembling some parameters into a hierarchy
    # (analysis parameters are in a flat hierarchy in XML files)
    node_frequent = ["no_fitting", "drift_correction", "threshold", "sigma", "foreground_sigma", "background_sigma",
                     "radius", "find_max_radius" ]

    node_camera = ["x_center", "x_start", "x_stop", "y_center", "y_start", "y_stop", "descriptor",
                   "max_frame", "pixel_size", "start_frame", "camera_gain", "camera_offset", "roi_size",
                   "static_background_estimate", "aoi_radius"]

    node_fitting = ["max_gap", "anscombe", "iterations", "peak_locations", "cutoff"]
    node_drift = ["frame_step", "d_scale"]
    node_z = ["wx_wo", "wx_c", "wx_d", "wxA", "wxB", "wxC", "wxD", "wy_wo", "wy_c", "wy_d",
              "wyA", "wyB", "wyC", "wyD", "z_value", "z_step", "min_z", "max_z", "z_correction", "do_zfit"]


    def __init__(self, **kwargs):
        super(Interface, self).__init__(**kwargs)
        self.parameters = params.Parameters()
        self.paramWidgets = {}
        self.seen_attrs = list()
        self.filename = None
        self._popup = None
        self.selectedFile = ""
        self.daopath = ""

        # Populate all the analysis tabs
        for p in Interface.panels:
            tabitem = TabbedPanelItem(text=p[10:])
            self.analysis_panel.add_widget(tabitem)
            sv = ScrollView(size=self.size, smooth_scroll_end=10)
            tabitem.add_widget(sv)

            treeview=self.makeTree(getattr(params, p)())
            sv.add_widget(treeview)

            # This is to select the first tab by default after initialization
            if p == "ParametersDAO":
                Clock.schedule_once(partial(self.switch, tabitem), 0)

        # Read previous DAOSTORM directory
        try:
            f = open("daopath.txt",'r')
            path = f.readline().rstrip()

            if os.path.exists(path + "/mufit_analysis.py"):
                self.daopath = path

            f.close()

        except:
            return


    def switch(self, tab, *args):
        self.analysis_panel.switch_to(tab)

    def handleSliderCurFrame(self, value):
        self.daxviewer.cur_frame = int(value)
        self.frame_number.text = str(self.daxviewer.cur_frame)
        self.daxviewer.displayFrame(True)

    def handleSliderContrast(self, value):
        self.daxviewer.displayFrame(True)

    def makeTree(self, parameters):
        tv=TreeView(hide_root=True)
        tv.size_hint_y = None
        tv.bind(minimum_height=tv.setter('height'))
        tv.add_node(TreeViewNode(orientation="horizontal", height=dp(12)))

        # These are needed for the widget to scroll
        # Also: the ScrollView can have only a single TreeView widget for this
        # to work

        # Pull parameters out of the different classes in Parameters
        #parameters = params.ParametersDAO()

        # Spinner box for the fit type
        if "model" not in self.paramWidgets.keys():
            attr="model"
            n = TreeViewNode(orientation="horizontal", size_hint_y=None, height=dp(24), padding=[18, 12])
            n.add_widget(Label(text=attr,  halign="left"))
            spinner = Spinner(text="Z", values = ['Z', '2dfixed', '2d', '3d'])
            spinner.name = attr
            spinner.nodetype = 'string'
            spinner.bind(text=self.handleTextChange)
            n.add_widget(spinner)
            self.paramWidgets[attr]=spinner
            tv.add_node(n)

            # Update current parameters
            self.updateParam(spinner)

        # Process some frequently-used params first
        for attr in Interface.node_frequent:
            if attr not in self.paramWidgets.keys():
                n = TreeViewNode(orientation="horizontal", size_hint_y=None, height=dp(24), padding=[18, 0])
                n.add_widget(Label(text=attr, halign="left"))

                # Make input of the right type (for now, TextInput and
                # CheckBox)
                if attr in Interface.checkboxes:
                    t = self.makeTreeNode(parameters.attr[attr], attr, "check")
                else:
                    t = self.makeTreeNode(parameters.attr[attr], attr, "textbox")

                self.paramWidgets[attr]=t
                n.add_widget(t)
                tv.add_node(n)

        # Some branches in the tree
        # camera, fitting, drift, z
        if isinstance(parameters, params.ParametersDAO):
            node_camera = tv.add_node(TreeViewLabel(text=' Camera'))
            node_fitting = tv.add_node(TreeViewLabel(text=' Fitting'))
            node_drift = tv.add_node(TreeViewLabel(text=' Drift'))
            node_z = tv.add_node(TreeViewLabel(text=' Z'))
            node_other = tv.add_node(TreeViewLabel(text=' Other'))

        # The rest of the parameters
        for attr in parameters.attr.keys():
            if attr not in self.paramWidgets.keys():
                n = TreeViewNode(orientation="horizontal", size_hint_y=None, height=dp(24), padding=[18, 0])
                n.add_widget(Label(text=attr, halign="left"))

                # Make input of the right type (for now, TextInput and
                # CheckBox)
                if attr in Interface.checkboxes:
                    t = self.makeTreeNode(parameters.attr[attr], attr, "check")
                else:
                    t = self.makeTreeNode(parameters.attr[attr], attr, "textbox")

                self.paramWidgets[attr]=t
                n.add_widget(t)

                # Assign to correct nodes in hierarchy
                if attr in Interface.node_camera:
                    tv.add_node(n, node_camera)
                elif attr in Interface.node_fitting:
                    tv.add_node(n, node_fitting)
                elif attr in Interface.node_drift:
                    tv.add_node(n, node_drift)
                elif attr in Interface.node_z:
                    tv.add_node(n, node_z)
                else:
                    if isinstance(parameters, params.ParametersDAO):
                        tv.add_node(n, node_other)
                    else:
                        tv.add_node(n) # Other tabs don't really need branches..

                #TreeWidget.seen_attrs.append(attr)

        return tv


    def makeTreeNode(self, parameter, widget_name, widget_type):
        if widget_type == "textbox":
            # Implement some input filteers
            if parameter[0] == 'int' or 'int' in parameter[0] and 'float' in parameter[0]:
                t = TextInput(text="", size_hint_x=.35, multiline=False, input_filter = 'int')

            elif parameter[0] == 'float':
                t = TextInput(text="", size_hint_x=.35, multiline=False, input_filter = 'float')

            else:
                t = TextInput(text="", size_hint_x=.35, multiline=False)

            # Avoid complaint about parameter that's (int, float)
            if 'int' in parameter[0] and 'float' in parameter[0]:
                t.nodetype = 'int'
            else:
                t.nodetype = parameter[0] #'string'


            # Bring up a file selection dialog if filename field is clicked
            #if t.nodetype == 'filename':
            #    t.bind(focus = self.handleSelectFilename)
            #else:
            #    t.bind(focus=self.handleUnfocus)
            if t.nodetype == 'filename':
                t.bind(on_double_tap=self.handleSelectFilename)

            t.bind(focus=self.handleUnfocus)


        else:
            t = CheckBox()
            t.nodetype = 'int'
            self.updateParam(t)
            t.bind(active=self.handleTextChange)

        t.name = widget_name
        return t


    def updateParam(self, instance):
        # Initialize parameter
        warnings=True

        if  isinstance(instance, CheckBox):
            value = instance.active
        else:
            value = instance.text

        if value:
            try:
                if not self.parameters.hasAttr(instance.name):
                    self.parameters.attr[instance.name] = [instance.nodetype, None]

                # Parse input and set according to parameters.initAttr
                # This is here to avoid creating an XML ElementTree

                if (instance.nodetype == "int"):
                    self.parameters.setAttr(instance.name, instance.nodetype, int(value), warnings)

                elif (instance.nodetype == "int-array"):
                    text_array = value.split(",")
                    int_array = []
                    for elt in text_array:
                        int_array.append(int(elt))
                        self.parameters.setAttr(instance.name, instance.nodetype, int_array, warnings)

                elif (instance.nodetype == "float"):
                    self.parameters.setAttr(instance.name, instance.nodetype, float(value), warnings)

                elif (instance.nodetype == "float-array"):
                    text_array = value.split(",")
                    float_array = []
                    for elt in text_array:
                        float_array.append(float(elt))
                        self.parameters.setAttr(instance.name, instance.nodetype, float_array, warnings)

                elif (instance.nodetype == "string-array"):
                    self.parameters.setAttr(instance.name, instance.nodetype, value.split(","), warnings)

                elif (instance.nodetype == "filename"):
                    #dirname = os.path.dirname(os.path.abspath(value))
                    #if not dirname:
                    #    dirname = os.getcwd()

                    if not os.path.isabs(value):
                        fname = os.path.join(self.directory, value)
                        if not isfile(fname):
                            return

                    else:
                        fname = value

                    self.parameters.setAttr(instance.name, instance.nodetype, fname, warnings)

                else: # everything else is assumed to be a string
                    self.parameters.setAttr(instance.name, instance.nodetype, value, warnings)

            except Exception as e:
                # Reset value of widget and return
                instance.text = ""
                return


    def handleUnfocus(self, instance, value):
        if not value: # Only on defocus or enter
            self.updateParam(instance)


    def handleTextChange(self, instance, value):
        self.updateParam(instance)

    def handleAnalyzeFrame(self):
        if self.parameters:
            self.daxviewer.parameters = self.parameters
            self.daxviewer.analyzeFrame()

    def handleAnalyzeDax(self):
        if self.parameters:
            self.daxviewer.parameters = self.parameters
            self.daxviewer.analyzeDax()

    def handleLoadZstring(self):
        content = StringDialog(load=self.loadZstring, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load Z calib string", content=content,
                                size_hint=(.3, .3), pos_hint= {"center_x":0.5,"center_y":.8})

        # Need this to focus the textinput when popup is opened
        self._popup.bind(on_open=self.on_popup_focus)
        self._popup.open()


    def on_popup_focus(self, popup):
        if popup:
            popup.content.stringinput.focus = True

    def loadZstring(self, zcalib):

        if self.parameters is None:
            self.parameters = params.ParametersDAO()

        props=re.findall('[A-Za-z0-9]+=', zcalib)
        parsed=re.split('[A-Za-z0-9]+=', zcalib)
        d = dict(zip(props, parsed[1:]))

        for key in d:
            p=key.strip('=')
            v=d[key].strip(' ;')

            if p in Interface.zfitvars.keys() and self.checkInput(v, 'float'):
                w = self.paramWidgets[Interface.zfitvars[p]]
                if float(v):
                   w.text = v
                   self.updateParam(w)

        self.dismiss_popup()

            #if p in self.zfitvars.keys() and self.checkInput(v, 'float'):
            #    self.parameters.setAttr(self.zfitvars[p], 'float', float(v))

        # Populate text boxes with parameters



    def checkInput(self, var, vartype):
        try:
            if vartype == 'int':
                int(var)

            elif vartype == 'float':
                float(var)

            return 1

        except:
            if var:
                self.error_dialog.showMessage('Invalid input!')

        return 0

    def handleLoadFile(self, filetype):
        if filetype == 'dax':
            content = LoadDialog(load=self.loadDax, cancel=self.dismiss_popup)
            content.filechooser.filters=['*.dax', '*.tif', '*.spe']
            if self.daxviewer.directory:
                content.filechooser.path = self.daxviewer.directory

            self._popup = Popup(title="Load dax", content=content,
                                size_hint=(1, 1), pos_hint= {"center_x":0.5,"center_y":.5})


        elif filetype == 'param':

            content = LoadDialog(load=self.loadParams, cancel=self.dismiss_popup)
            content.filechooser.filters=['*.xml']

            if self.daxviewer.directory:
                content.filechooser.path = self.daxviewer.directory

            self._popup = Popup(title="Load params", content=content,
                                size_hint=(1, 1), pos_hint= {"center_x":0.5,"center_y":.5})


        elif filetype == 'mol':
            content = LoadDialog(load=self.loadMList, cancel=self.dismiss_popup)
            content.filechooser.filters=['*.h5', '*.hdf5', '*.hdf', '*.bin']
            if self.daxviewer.directory:
                content.filechooser.path = self.daxviewer.directory

            self._popup = Popup(title="Load mol list", content=content,
                                size_hint=(1, 1), pos_hint= {"center_x":0.5,"center_y":.5})

        elif filetype == 'dao':
            content = LoadDialog(load=self.setDaoPath, cancel=self.dismiss_popup)
            content.filechooser.filters=[lambda folder, filename: not filename.endswith('')]

            self._popup = Popup(title="Set DAOstorm path", content=content,
                                size_hint=(1, 1), pos_hint= {"center_x":0.5,"center_y":.5})


        elif filetype == 'dir':
            content = LoadDialog(load=self.handleBatchAnalyze, cancel=self.dismiss_popup)
            content.filechooser.filters=[lambda folder, filename: not filename.endswith('')]

            if self.daxviewer.directory:
                content.filechooser.path = self.daxviewer.directory

            self._popup = Popup(title="Select directory", content=content,
                                size_hint=(1, 1), pos_hint= {"center_x":0.5,"center_y":.5})

        else:
            content = LoadDialog(load=self.selectFilename, cancel=self.dismiss_popup)
            if self.daxviewer.directory:
                content.filechooser.path = self.daxviewer.directory

            self._popup = Popup(title="Select file", content=content,
                                size_hint=(1, 1), pos_hint= {"center_x":0.5,"center_y":.5})

        self._popup.open()


    def handleBatchAnalyze(self, path, filename):
        self.dismiss_popup()

        if not (self.parameters.hasAttr("threshold") and self.daopath):
            return

        self.daxviewer.parameters = self.parameters
        self.daxviewer.daopath = self.daopath
        self.daxviewer.batchDaoBackground(path)


    def setDaoPath(self, path, filename):
        if os.path.exists(path + "/mufit_analysis.py"):
            self.daopath = path

            try:
                f=open('daopath.txt', 'w')
                f.write(self.daopath + "\n")
                f.close()
            except:
                pass

            self.dismiss_popup()

        else:
            self.dismiss_popup()
            content = PopupDialog(ok=self.dismiss_popup)
            content.messagelabel.text = "DAOstorm path not correct"
            self._popup = Popup(title="DAOstorm path", content=content,
                                size_hint=(.3, .3), pos_hint= {"center_x":0.5,"center_y":.5})

            self._popup.open()



    def selectFilename(self, path, filename):
        self.selectedFile.text = os.path.join(path, filename)
        self.updateParam(self.selectedFile)
        self.dismiss_popup()


    def handleSelectFilename(self, instance):
        self.selectedFile = instance
        self.handleLoadFile('filename')


    def handleSaveParams(self):
        content = SaveDialog(save=self.saveParams, cancel=self.dismiss_popup)
        content.filechooser.filters=['*.xml']

        if self.daxviewer.directory:
            content.filechooser.path = self.daxviewer.directory

        self._popup = Popup(title="Save params", content=content,
                            size_hint=(1, 1), pos_hint= {"center_x":0.5,"center_y":.5})

        self._popup.open()


    def saveParams(self, path, filename):
        if not self.parameters:
            return

        if filename:
            self.parameters.toXMLFile(os.path.join(path,filename))

        self.dismiss_popup()


    def dismiss_popup(self):
        self._popup.dismiss()


    def loadParams(self, path, filename):

        file_path = os.path.join(path, filename)

        # Try different Parameters classes until we can accommodate
        # all the parameters read in
        for p in Interface.panels:
            try:
                self.parameters = getattr(params, p)()
                self.parameters.initFromFile(file_path)
                break
            except:
                pass

        # Populate text boxes with data from the file
        for i in self.parameters.attr.keys():

            if self.parameters.hasAttr(i) and i != 'parameters_file':
                w = self.paramWidgets[i]
                value = self.parameters.getAttr(i)

                if isinstance(w, TextInput) or isinstance(w, Spinner):
                    w.text = str(value)
                else:
                    w.active = int(value)

        if self._popup:
            self.dismiss_popup()


    def loadMList(self, path, filename):
        list_filename = os.path.join(path, filename)

        if self.daxviewer.locs1_list is not None:
            self.daxviewer.cleanDrawing()

        self.daxviewer.directory = os.path.dirname(list_filename)

        try:
            if saH5Py.isSAHDF5(list_filename):
                self.daxviewer.locs1_list = MoleculeListHDF5(filename = list_filename, parent=self.daxviewer)
            else:
                self.daxviewer.locs1_list = MoleculeListI3(filename = list_filename, parent=self.daxviewer)
        except:
            print("Couldn't load mol list")

        self.daxviewer.redrawFits(None, None)
        if self._popup:
            self.dismiss_popup()


    def loadDax(self, path, filename):
        try:
            self.daxviewer.addDax(os.path.join(path, filename))
        except:
            print("Something went wrong")

        if self._popup:
            self.dismiss_popup()


# Classes for popups
class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class PopupDialog(FloatLayout):
    ok = ObjectProperty(None)

class StringDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

# Container class for Tree nodes
class TreeViewNode(BoxLayout, TreeViewNode):
    pass

class Analyzer(App):
    #daxviewer = ObjectProperty(None)

    def build(self):
        root = Interface()
        Window.bind(on_dropfile=self.on_file_drop)
        return root


    def on_file_drop(self, window, file_path):
        splitpath = os.path.split(file_path.decode())
        splitfile = os.path.splitext(file_path.decode())

        if splitfile[1] == '.dax':
            self.root.loadDax(splitpath[0], splitpath[1])

        elif splitfile[1] == '.xml':
            self.root.loadParams(splitpath[0], splitpath[1])

        elif splitfile[1] == '.h5' or splitfile[1] == '.hdf' or splitfile[1] == '.hdf5' or splitfile[1] == '.bin':
            self.root.loadMList(splitpath[0], splitpath[1])


class MoleculeList(object):
    """
    Handle molecule list.
    """
    def __init__(self, parent,  **kwds):
        super(MoleculeList, self).__init__(**kwds)

        self.fields = []
        self.last_frame = -1
        self.last_index = 0
        self.locs = {}
        self.mol_items = []
        self.reader = None
        self.parent = parent

    def createMolItems(self, frame_number, nm_per_pixel):

        # Only load new localizations if this is a different frame.
        if (frame_number != self.last_frame):
            self.last_frame = frame_number
            self.loadLocalizations(frame_number, nm_per_pixel)
            self.last_index = 0

        self.mol_items = []
        if bool(self.locs):
            if "xsigma" in self.locs:
                xs = self.locs["xsigma"]
                ys = self.locs["ysigma"]
            else:
                xs = 1.5*numpy.ones(self.locs["x"].size)
                ys = 1.5*numpy.ones(self.locs["x"].size)

            imgstretch = self.parent.image.norm_image_size[0]/self.parent.image.texture_size[0]
            ytransform = (self.parent.size[1] - self.parent.image.norm_image_size[1])/2

            # Add ellipse objects
            for i in range(self.locs["x"].size):
                ig = InstructionGroup()
                ig.add(Color(0,1,0))

                x = imgstretch*(self.locs["x"][i] +1 - 0.5*xs[i]*6.0 - 0.5)
                y = imgstretch*(self.locs["y"][i] +1 - 0.5*ys[i]*6.0 - 0.5) + ytransform

                ig.add( Line( width=1.5, ellipse = (x, y, xs[i]*6.0*imgstretch, ys[i]*6.0*imgstretch)))

                self.mol_items.append(ig)

                #self.mol_items.append(MoleculeItem(self.locs["x"][i] + 1,
                #                                   self.locs["y"][i] + 1,
                #                                   xs[i]*6.0,
                #                                   ys[i]*6.0,
                #                                   self.mtype))

        return self.mol_items

    def loadLocalizations(self, frame_number, nm_per_pixel):
        pass

    def getClosest(self, px, py):
        """
        Return a dictionary with information about the closest localization
        to px, py.
        """
        vals = {}
        if bool(self.locs) and (self.locs["x"].size > 0):

            # Find the one nearest to px, py.
            dx = self.locs["x"] - px - 0.5
            dy = self.locs["y"] - py - 0.5
            dist = dx*dx+dy*dy
            closest_index = numpy.argmin(dist)

            # Unmark old item, mark new item
            self.mol_items[self.last_index].setMarked(False)
            self.mol_items[closest_index].setMarked(True)
            self.last_index = closest_index

            # Create a dictionary containing the data for this molecule.
            vals = {}
            for field in self.locs:
                vals[field] = self.locs[field][closest_index]

        return vals

    def getFields(self):
        return self.fields


# Simple subclass of MoleculeList to handle single-frame localizations (for testing fit parameters)
class MoleculeListSingle(MoleculeList):
    def __init__(self, locs, lastframe, parent, **kwds):
        super(MoleculeListSingle, self).__init__(parent, **kwds)
        self.fields = ["x",
                       "y",
                       "z",
                       "background",
                       "error",
                       "height",
                       "sum",
                       "xsigma",
                       "ysigma",
                       "category",
                       "iterations",
                       "significance"]

        if ("xsigma" in locs) and (not "ysigma" in locs):
            locs["ysigma"] = locs["xsigma"]

        self.locs = locs
        self.last_frame = lastframe
        #self.parent = parent


    def cleanUp(self):
        return

    def createMolItems(self, frame_number, nm_per_pixel):
        # Clear out localizations if we're on a different frame
        if (frame_number != self.last_frame):
            self.locs = None

        self.mol_items = []
        if bool(self.locs):
            if "xsigma" in self.locs:
                xs = self.locs["xsigma"]
                ys = self.locs["ysigma"]
            else:
                xs = 1.5*numpy.ones(self.locs["x"].size)
                ys = 1.5*numpy.ones(self.locs["x"].size)

            imgstretch = self.parent.image.norm_image_size[0]/self.parent.image.texture_size[0]

            #print(self.parent.image.norm_image_size)
            #print(self.parent.size)
            #print(self.parent.bbox)
            #ytransform = (self.parent.bbox[1][1] - self.parent.image.norm_image_size[1])/2

            ytransform = (self.parent.size[1] - self.parent.image.norm_image_size[1])/2

            for i in range(self.locs["x"].size):
                ig = InstructionGroup()
                ig.add(Color(0,1,0))

                x = imgstretch*(self.locs["x"][i] +1 - 0.5*xs[i]*6.0 - 0.5)
                y = imgstretch*(self.locs["y"][i] +1 - 0.5*ys[i]*6.0 - 0.5) + ytransform

                ig.add( Line( width=1.5, ellipse = (x, y, xs[i]*6.0*imgstretch, ys[i]*6.0*imgstretch)))
                self.mol_items.append(ig)

                #self.mol_items.append(MoleculeItem(self.locs["x"][i] + 1,
                #                                   self.locs["y"][i] + 1,
                #                                   xs[i]*6.0,
                #                                   ys[i]*6.0))

        return self.mol_items


class MoleculeListHDF5(MoleculeList):
    """
    Handle HDF5 molecule list.
    """
    def __init__(self, filename, parent, **kwds):
        super(MoleculeListHDF5, self).__init__(parent, **kwds)

        self.fields = ["x",
                       "y",
                       "z",
                       "background",
                       "error",
                       "height",
                       "sum",
                       "xsigma",
                       "ysigma",
                       "category",
                       "iterations",
                       "significance"]

        self.reader = saH5Py.SAH5Py(filename)

    def cleanUp(self):
        self.reader.close(verbose = False)

    def loadLocalizations(self, frame_number, nm_per_pixel):
        self.locs = self.reader.getLocalizationsInFrame(frame_number)
        if bool(self.locs):
#            if not "xsigma" in locs:
#                locs["xsigma"] = 1.5*numpy.ones(locs["x"].size)
#                locs["ysigma"] = 1.5*numpy.ones(locs["x"].size)
            if ("xsigma" in self.locs) and (not "ysigma" in self.locs):
                self.locs["ysigma"] = self.locs["xsigma"]
        #return locs


class MoleculeListI3(MoleculeList):
    """
    Handle Insight3 molecule list.
    """
    def __init__(self, filename, parent, **kwds):
        super(MoleculeListI3, self).__init__(parent, **kwds)

        self.fields = ["x",
                       "y",
                       "z",
                       "background",
                       "error",
                       "height",
                       "sum",
                       "xsigma",
                       "ysigma"]

        self.reader = readinsight3.I3Reader(filename)
        self.n_locs = self.reader.getNumberMolecules()

    def cleanUp(self):
        self.reader.close()

    def loadLocalizations(self, frame_number, nm_per_pixel):
        if (self.n_locs > 0):
            fnum = frame_number + 1
            i3data = self.reader.getMoleculesInFrame(fnum)
            return i3dtype.convertToSAHDF5(i3data, fnum, nm_per_pixel)
        else:
            return {}


Analyzer().run()
