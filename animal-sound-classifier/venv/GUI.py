import os
import numpy as np
from kivy import Config
from kivy.app import App
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.uix import popup
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty, StringProperty, BooleanProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from main import execute
from train_test_create import Create
from functions import playAudio, plotAudio, initiate_birds, initiate_libr, list_view
from kivy.core.window import Window
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior

output = list()
importedfiles = list()
class_param = {'RF': [],'SVM': []}
clust_param = {'KMeans': [],'DBSCAN': []}
is_finished = True
is_rf = False
is_svm = False
is_kmeans = False
is_dbscan = False
selectedfile = ''
selectedfile2 = ''
temp = 'abc'
temp2 = 'abc'
is_selected_for_graph = False

def set_temp(selectedfile):
    global temp
    temp = selectedfile

def set_temp2(selectedfile2):
    global temp2
    temp2 = selectedfile2

class UploadWindow(Screen):
    paths = StringProperty()

    def show_load_list(self):
        content = LoadDialog(load=self.load_list, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load the sound files", content=content, size_hint=(.5, .5))
        self._popup.open()

    def load_list(self, path, filename):
#        with open(os.path.join(path, filename[0])) as stream:
#            self.inputpath.text = stream.read()
        filename_actual = os.path.basename(filename.pop())
        global importedfiles
        fileimported = list()
        fileimported.append(filename_actual)
        fileimported.append(path)
        importedfiles.append(fileimported)

        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

class LabelledWindow(Screen):
    next_page = StringProperty("class-spec")
    detail_page = StringProperty("lbldetails")

class ClassificationSpecWindow(Screen):
    num_estimator = float()
    optimized1 = False
    five_fold1_state = False
    rf_state = False
    svm_state = False
    c_val = float()
    optimized2 = False
    kernel = StringProperty("linear")
    five_fold2_state = False

    def estimator_entered(self, value):
        self.num_estimator = value

    def optimized1_active(self, state):
        if state:
            self.optimized1 = True
        else:
            self.optimized1 = False

    def five_fold1_checkbox_active(self, state):
        if state:
            self.five_fold1_state = True
        else:
            self.five_fold1_state = False

    def five_fold2_checkbox_active(self, state):
        if state:
            self.five_fold2_state = True
        else:
            self.five_fold2_state = False

    def insert_check1(self, state):
        if state:
            self.rf_state = True
        else:
            self.rf_state = False

    def insert_check2(self, state):
        if state:
            self.svm_state = True
        else:
            self.svm_state = False

    def c_entered(self, value):
        self.c_val = value

    def spinner_clicked(self, value):
        self.kernel = value

    def optimized2_active(self, state):
        if state:
            self.optimized2 = True
        else:
            self.optimized2 = False

    def insert_data(self):
        output.append(True)

        if self.rf_state:
            if self.optimized1:
                class_param.get('RF').append(self.optimized1)
                class_param.get('RF').append(self.five_fold1_state)
            else:
                class_param.get('RF').append(self.num_estimator)
                class_param.get('RF').append(self.optimized1)
                class_param.get('RF').append(self.five_fold1_state)
        else:
            pass

        if self.svm_state:
            if self.optimized2:
                class_param.get('SVM').append(self.optimized2)
                class_param.get('SVM').append(self.five_fold2_state)
            else:
                class_param.get('SVM').append(self.c_val)
                class_param.get('SVM').append(self.kernel)
                class_param.get('SVM').append(self.optimized2)
                class_param.get('SVM').append(self.five_fold2_state)
        else:
            pass

        global is_rf
        is_rf = self.rf_state
        global is_svm
        is_svm = self.svm_state

        output.append(class_param)

        if self.rf_state and not self.svm_state:
            rftext = open("RF.txt", "w+")
            label1list = execute(output)

            for i in label1list:
                rftext.write(i)
            rftext.close()



        if self.svm_state and not self.rf_state:
            svmtext = open("SVM.txt", "w+")
            label2list = execute(output)

            for i in label2list:
                svmtext.write(i)
            svmtext.close()

        if self.rf_state and self.svm_state:
            rftext = open("RF.txt", "w+")
            svmtext = open("SVM.txt", "w+")
            labellist = execute(output)

            if self.five_fold1_state and not self.five_fold2_state:
                label1list = labellist[:12]
                label2list = labellist[12:]
            if self.five_fold2_state and not self.five_fold1_state:
                label1list = labellist[:5]
                label2list = labellist[5:]
            if self.five_fold2_state and self.five_fold1_state:
                label1list = labellist[:12]
                label2list = labellist[12:]
            for i in label1list:
                rftext.write(i)
            rftext.close()

            for i in label2list:
                svmtext.write(i)
            svmtext.close()

class ClusteringSpecWindow(Screen):
    num_clusters = float()
    optimized1 = False
    kmeans_state = False
    dbscan_state = False
    eps = float()
    min_samples = float()
    optimized2 = False

    def cluster_entered(self, value):
        self.num_clusters = value

    def optimized1_active(self, state):
        if state:
            self.optimized1 = True
        else:
            self.optimized1 = False

    def insert_check1(self, state):
        if state:
            self.kmeans_state = True
        else:
            self.kmeans_state = False

    def insert_check2(self, state):
        if state:
            self.dbscan_state = True
        else:
            self.dbscan_state = False

    def eps_entered(self, value):
        self.eps = value

    def min_samples_entered(self, value):
        self.min_samples = value

    def optimized2_active(self, state):
        if state:
            self.optimized2 = True
        else:
            self.optimized2 = False

    def insert_data(self):
        output.append(False)

        if self.kmeans_state:
            if self.optimized1:
                clust_param.get('KMeans').append(self.optimized1)
            else:
                clust_param.get('KMeans').append(self.num_clusters)
                clust_param.get('KMeans').append(self.optimized1)
        else:
            pass

        if self.dbscan_state:
            if self.optimized2:
                clust_param.get('DBSCAN').append(self.optimized2)
            else:
                clust_param.get('DBSCAN').append(self.eps)
                clust_param.get('DBSCAN').append(self.min_samples)
                clust_param.get('DBSCAN').append(self.optimized2)
        else:
            pass

        global is_kmeans
        is_kmeans = self.kmeans_state
        global is_dbscan
        is_dbscan = self.dbscan_state

        output.append(clust_param)

        if self.kmeans_state and not self.dbscan_state:
            kmeanstext = open("KMeans.txt", "w+")
            label1list = execute(output)

            for i in label1list:
                kmeanstext.write(i)
            kmeanstext.close()

        if self.dbscan_state and not self.kmeans_state:
            dbscantext = open("DBSCAN.txt", "w+")
            label1list = execute(output)

            for i in label1list:
                dbscantext.write(i)
            dbscantext.close()

        if self.dbscan_state and self.kmeans_state:
            kmeanstext = open("KMeans.txt", "w+")
            dbscantext = open("DBSCAN.txt", "w+")
            labellist = execute(output)
            label1list = labellist[0]
            label2list = labellist[1:]

            for i in label1list:
                kmeanstext.write(i)
            kmeanstext.close()

            for i in label2list:
                dbscantext.write(i)
            dbscantext.close()


class WaitScreenClass(Screen):
    label = StringProperty("")

class WaitScreenClust(Screen):
    label = StringProperty("")

class RFResultsWindow(Screen):
    label = StringProperty("")
    graphpath = StringProperty("black.jpg")

    def on_pre_enter(self):
        if is_rf:
            self.graphpath = "RF-CM.png"
            with open("RF.txt") as f:
                contents = f.read()
                self.label = contents

class SVMResultsWindow(Screen):
    label = StringProperty("")
    graphpath = StringProperty("black.jpg")

    def on_pre_enter(self):
        if is_svm:
            self.graphpath = "SVM-CM.png"
            with open("SVM.txt") as f:
                contents = f.read()
                self.label = contents

class KMeansResultsWindow(Screen):
    label = StringProperty("")
    graphpath = StringProperty("black.jpg")

    def on_pre_enter(self):
        if is_kmeans:
            self.graphpath = "K-Means.png"
            with open("KMeans.txt") as f:
                contents = f.read()
                self.label = contents

class DBSCANResultsWindow(Screen):
    label = StringProperty("")
    graphpath = StringProperty("black.jpg")

    def on_pre_enter(self):
        if is_dbscan:
            self.graphpath = "DBSCAN.png"
            with open("DBSCAN.txt") as f:
                contents = f.read()
                self.label = contents

class LabelledDetailsWindow(Screen):
    graphpath = StringProperty("black.jpg")
    path = os.getcwd() + "/bird-sounds/"
    global temp

    def play(self):
        playAudio(temp)

    def plot(self):
        plotAudio(temp)
        self.graphpath = "frequency.png"

    def delete_plot(self):
        if os.path.exists("frequency.png"):
            os.remove("frequency.png")
        self.graphpath = "black.jpg"

class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    ''' Adds selection and focus behaviour to the view. '''

class SelectableLabel(RecycleDataViewBehavior, Label):
    ''' Add selection support to the Label '''
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        ''' Catch and handle the view changes '''
        self.index = index
        return super(SelectableLabel, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        ''' Add selection on touch down '''
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        '''Respond to the selection of items in the view. '''
        self.selected = is_selected
        if is_selected:
            #print("selection changed to {0}".format(rv.data[index]))
            global selectedfile
            selectedfile = rv.data[index].get('text')
            birds = initiate_birds()
            filename = selectedfile
            path = os.getcwd() + "/bird-sounds/"
            for x in birds:
                if x[:4] == filename[:4]:
                    path = path + x +'/' + filename
                    set_temp(path)

        else:
            print("selection removed for {0}".format(rv.data[index]))

class RV(RecycleView):
    # Initials
    path = os.getcwd() + "/"
    song = os.getcwd() + "/bird-sounds/"
    bird_path = path + "bird-dir/bird-types.txt"
    bird_names = open(bird_path, "r")

    # Birds List
    birds = initiate_birds()

    # Birds and their songs Dictionary
    libr = initiate_libr(birds)

    def __init__(self, **kwargs):
        super(RV, self).__init__(**kwargs)
        tmp = list()
        for x in self.libr.keys():
            tmp.append(str(f'--{x}--'))
            for y in self.libr.get(x):
                tmp.append(y)
        self.data = [{'text': str(x)} for x in tmp]

class UnlabelledDetailsWindow(Screen):
    graphpath = StringProperty("black.jpg")
    path = os.getcwd() + "/bird-sounds/"
    global temp2

    def play(self):
        playAudio(temp2)

    def plot(self):
        plotAudio(temp2)
        self.graphpath = "frequency.png"

    def delete_plot(self):
        if os.path.exists("frequency.png"):
            os.remove("frequency.png")
        self.graphpath = "black.jpg"

class Un_SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    ''' Adds selection and focus behaviour to the view. '''

class Un_SelectableLabel(RecycleDataViewBehavior, Label):
    ''' Add selection support to the Label '''
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        ''' Catch and handle the view changes '''
        self.index = index
        return super(Un_SelectableLabel, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        ''' Add selection on touch down '''
        if super(Un_SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        '''Respond to the selection of items in the view. '''
        self.selected = is_selected
        if is_selected:
            #print("selection changed to {0}".format(rv.data[index]))
            global selectedfile2
            selectedfile2 = rv.data[index].get('text')
            X_train, names = list_view()
            filename = selectedfile2
            path2 = X_train[names.index(filename)]
            set_temp2(path2)
        else:
            print("selection removed for {0}".format(rv.data[index]))

class Un_RV(RecycleView):
    # Initials

    path = os.getcwd() + "/cats_dogs/"

    X_train, names = list_view()
    def __init__(self, **kwargs):
        super(Un_RV, self).__init__(**kwargs)

        self.data = [{'text': str(x)} for x in self.names]

class HistoryWindow(Screen):
    def reset(self):
        self.location.text = ""

    def historyBtn(self):
        self.reset()

    def homepageBtn(self):
        self.reset()
        sm.current = "upload"

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("Windows.kv")

sm = WindowManager()

screens = [UploadWindow(name="upload"), RFResultsWindow(name="rfresults"), SVMResultsWindow(name="svmresults"),
           KMeansResultsWindow(name="kmeansresults"), DBSCANResultsWindow(name="dbscanresults"),
           LabelledDetailsWindow(name="lbldetails"), UnlabelledDetailsWindow(name="unlbldetails"),
           HistoryWindow(name="history"), LabelledWindow(name="labelledwin"),
           ClassificationSpecWindow(name="class-spec"), ClusteringSpecWindow(name="clust-spec"),
           WaitScreenClass(name="waitscreenclass"), WaitScreenClust(name="waitscreenclust")]
for screen in screens:
    sm.add_widget(screen)

sm.current = "labelledwin"


class MyMainApp(App):
    def build(self):
        return sm


def run():
    MyMainApp().run()

def get_output():
    return output



if __name__ == "__main__":
    Config.set('graphics', 'window_state', 'maximized')
    Config.write()
    Create()
    run()






'''
if __name__ == "__main__":
    MyMainApp().run()
    get_output()
'''

'''
class AlgoWindow(Screen):
    check1 = StringProperty("RF")
    check2 = StringProperty("SVM")
    check1output = StringProperty("")
    check2output = StringProperty("")
    check1label = StringProperty("Random Forest")
    check2label = StringProperty("SVM")
    param1_1 = StringProperty("# Estimators")

    def activateclassification(self):
        self.check1="RF"
        self.check1label="Random Forest"
        self.param1_1="# Estimators"
        self.check2="SVM"
        self.check2label="SVM"

    def activateclustering(self):
        self.check1="KMeans"
        self.check1label="K-Means"
        self.param1_1="# Clusters"
        self.check2="DBSCAN"
        self.check2label="DBSCAN"

    def insert_check1(self):
        if True:
            output.append(self.check1output)
        else:
            pass

    def insert_check2(self):
        if True:
            output.append(self.check2output)
        else:
            pass

    def insert_data(self):
        self.insert_check1()
        self.insert_check2()
        output.append(True)
'''
