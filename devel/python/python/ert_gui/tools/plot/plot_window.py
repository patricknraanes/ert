from PyQt4.QtCore import Qt
from PyQt4.QtGui import QMainWindow, QDockWidget, QTabWidget, QWidget, QVBoxLayout
from ert.util import CTime
from ert_gui.models.connectors.init import CaseSelectorModel
from ert_gui.tools.plot import PlotPanel, DataTypeKeysWidget, CaseSelectionWidget, ExportPlot, CustomizePlotWidget, PlotToolBar, PlotMetricsTracker, PlotPanelTracker
from ert_gui.tools.plot.data import PlotDataFetcher
from ert_gui.widgets.util import may_take_a_long_time


class PlotWindow(QMainWindow):
    def __init__(self, parent):
        QMainWindow.__init__(self, parent)

        self.setMinimumWidth(750)
        self.setMinimumHeight(500)

        self.setWindowTitle("Plotting")
        self.activateWindow()

        self.__plot_data = None

        self.__plot_metrics_tracker = PlotMetricsTracker()
        self.__plot_metrics_tracker.addScaleType("value", float)
        self.__plot_metrics_tracker.addScaleType("depth", float)
        self.__plot_metrics_tracker.addScaleType("pca", float)
        self.__plot_metrics_tracker.addScaleType("index", int)
        self.__plot_metrics_tracker.addScaleType("count", int)
        self.__plot_metrics_tracker.addScaleType("time", CTime)

        self.__central_tab = QTabWidget()
        self.__central_tab.currentChanged.connect(self.currentPlotChanged)

        self.__plot_panel_tracker = PlotPanelTracker(self.__central_tab)
        self.__plot_panel_tracker.addKeyTypeTester("summary", PlotDataFetcher.isSummaryKey)
        self.__plot_panel_tracker.addKeyTypeTester("block", PlotDataFetcher.isBlockObservationKey)
        self.__plot_panel_tracker.addKeyTypeTester("gen_kw", PlotDataFetcher.isGenKWKey)
        self.__plot_panel_tracker.addKeyTypeTester("gen_data", PlotDataFetcher.isGenDataKey)
        self.__plot_panel_tracker.addKeyTypeTester("custom_pca", PlotDataFetcher.isCustomPcaDataKey)


        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(central_layout)

        self.__toolbar = PlotToolBar()

        central_layout.addWidget(self.__toolbar)
        central_layout.addWidget(self.__central_tab)

        self.setCentralWidget(central_widget)


        self.__plot_panels = []
        """:type: list of PlotPanel"""

        self.addPlotPanel("Ensemble plot", "gui/plots/simple_plot.html", short_name="Plot")
        self.addPlotPanel("Ensemble overview plot", "gui/plots/simple_overview_plot.html", short_name="oPlot")
        self.addPlotPanel("Histogram", "gui/plots/histogram.html", short_name="Histogram")
        self.addPlotPanel("Distribution", "gui/plots/gen_kw.html", short_name="Distribution")
        self.addPlotPanel("RFT plot", "gui/plots/rft.html", short_name="RFT")
        self.addPlotPanel("RFT overview plot", "gui/plots/rft_overview.html", short_name="oRFT")
        self.addPlotPanel("Ensemble plot", "gui/plots/gen_data.html", short_name="epGenData")
        self.addPlotPanel("Ensemble overview plot", "gui/plots/gen_data_overview.html", short_name="eopGenData")
        self.addPlotPanel("PCA plot", "gui/plots/pca.html", short_name="PCA")

        self.__data_type_keys_widget = DataTypeKeysWidget()
        self.__data_type_keys_widget.dataTypeKeySelected.connect(self.keySelected)
        self.addDock("Data types", self.__data_type_keys_widget)

        current_case = CaseSelectorModel().getCurrentChoice()
        self.__case_selection_widget = CaseSelectionWidget(current_case)
        self.__case_selection_widget.caseSelectionChanged.connect(self.caseSelectionChanged)
        plot_case_dock = self.addDock("Plot case", self.__case_selection_widget)

        self.__customize_plot_widget = CustomizePlotWidget()
        self.__customize_plot_widget.customPlotSettingsChanged.connect(self.plotSettingsChanged)
        customize_plot_dock = self.addDock("Customize", self.__customize_plot_widget)

        self.__toolbar.exportClicked.connect(self.exportActivePlot)
        self.__toolbar.plotScalesChanged.connect(self.plotSettingsChanged)
        self.__toolbar.reportStepChanged.connect(self.plotSettingsChanged)

        self.__exporter = None
        self.tabifyDockWidget(plot_case_dock, customize_plot_dock)

        plot_case_dock.show()
        plot_case_dock.raise_()

        self.__plot_cases = self.__case_selection_widget.getPlotCaseNames()


    def getActivePlot(self):
        """ @rtype: PlotPanel """
        if not self.__central_tab.currentIndex() > -1:
            raise AssertionError("No plot selected!")

        active_plot =  self.__central_tab.currentWidget()
        assert isinstance(active_plot, PlotPanel)

        return active_plot

    def currentPlotChanged(self):
        active_plot = self.getActivePlot()

        if active_plot.isReady():
            x_axis_type_name = active_plot.xAxisType()
            x_axis_type = self.__plot_metrics_tracker.getType(x_axis_type_name)

            y_axis_type_name = active_plot.yAxisType()
            y_axis_type = self.__plot_metrics_tracker.getType(y_axis_type_name)

            x_min, x_max = self.__plot_metrics_tracker.getScalesForType(x_axis_type_name)
            y_min, y_max = self.__plot_metrics_tracker.getScalesForType(y_axis_type_name)

            self.__toolbar.setToolBarOptions(x_axis_type, y_axis_type, active_plot.isReportStepCapable())
            self.__toolbar.setScales(x_min, x_max, y_min, y_max)


    def plotSettingsChanged(self):
        x_min, x_max = self.__toolbar.getXScales()
        y_min, y_max = self.__toolbar.getYScales()

        active_plot = self.getActivePlot()

        x_axis_type = active_plot.xAxisType()
        y_axis_type = active_plot.yAxisType()

        self.__plot_metrics_tracker.setScalesForType(x_axis_type, x_min, x_max)
        self.__plot_metrics_tracker.setScalesForType(y_axis_type, y_min, y_max)

        self.updatePlots()


    def updatePlots(self):
        report_step = self.__toolbar.getReportStep()

        for plot_panel in self.__plot_panels:
            if plot_panel.isPlotVisible():
                model = plot_panel.getPlotBridge()
                model.setPlotData(self.__plot_data)
                model.setCustomSettings(self.__customize_plot_widget.getCustomSettings())
                model.setReportStepTime(report_step)

                x_axis_type_name = plot_panel.xAxisType()
                y_axis_type_name = plot_panel.yAxisType()

                x_min, x_max = self.__plot_metrics_tracker.getScalesForType(x_axis_type_name)
                y_min, y_max = self.__plot_metrics_tracker.getScalesForType(y_axis_type_name)

                model.setScales(x_min, x_max, y_min, y_max)

                plot_panel.renderNow()


    def exportActivePlot(self):
        active_plot = self.getActivePlot()

        if self.__exporter is None:
            path = None
        else:
            path = self.__exporter.getCurrentPath()


        report_step = self.__toolbar.getReportStep()

        x_axis_type_name = active_plot.xAxisType()
        y_axis_type_name = active_plot.yAxisType()

        x_min, x_max = self.__plot_metrics_tracker.getScalesForType(x_axis_type_name)
        y_min, y_max = self.__plot_metrics_tracker.getScalesForType(y_axis_type_name)

        settings = {"x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "report_step": report_step}

        self.__exporter = ExportPlot(active_plot, settings, self.__customize_plot_widget.getCustomSettings(), path)

        self.__exporter.export()


    def addPlotPanel(self, name, path, short_name=None):
        if short_name is None:
            short_name = name

        plot_panel = PlotPanel(name, short_name, path)
        plot_panel.plotReady.connect(self.plotReady)
        self.__plot_panels.append(plot_panel)
        self.__central_tab.addTab(plot_panel, name)


    def addDock(self, name, widget, area=Qt.LeftDockWidgetArea, allowed_areas=Qt.AllDockWidgetAreas):
        dock_widget = QDockWidget(name)
        dock_widget.setObjectName("%sDock" % name)
        dock_widget.setWidget(widget)
        dock_widget.setAllowedAreas(allowed_areas)
        dock_widget.setFeatures(QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)

        self.addDockWidget(area, dock_widget)
        return dock_widget


    def checkPlotStatus(self):
        for plot_panel in self.__plot_panels:
            if not plot_panel.isReady():
                return False

        if len(self.__plot_cases) == 0:
            return False

        return True

    def plotReady(self):
        if self.checkPlotStatus():
            self.__data_type_keys_widget.selectDefault()
            self.currentPlotChanged()
            self.__customize_plot_widget.emitChange()


    def caseSelectionChanged(self):
        self.__plot_cases = self.__case_selection_widget.getPlotCaseNames()
        self.keySelected(self.__plot_metrics_tracker.getDataTypeKey())


    def showOrHidePlotTab(self, plot_panel, is_visible, show_plot):
        plot_panel.setPlotIsVisible(show_plot)
        if show_plot and not is_visible:
            index = self.__plot_panels.index(plot_panel)
            self.__central_tab.insertTab(index, plot_panel, plot_panel.getName())
        elif not show_plot and is_visible:
            index = self.__central_tab.indexOf(plot_panel)
            self.__central_tab.removeTab(index)


    @may_take_a_long_time
    def keySelected(self, key):
        key = str(key)
        old_data_type_key = self.__plot_metrics_tracker.getDataTypeKey()
        self.__plot_metrics_tracker.setDataTypeKey(key)

        plot_data_fetcher = PlotDataFetcher()
        self.__plot_data = plot_data_fetcher.getPlotDataForKeyAndCases(key, self.__plot_cases)
        self.__plot_data.setParent(self)

        self.__central_tab.blockSignals(True)

        self.__plot_panel_tracker.storePlotType(plot_data_fetcher, old_data_type_key)

        for plot_panel in self.__plot_panels:
            self.showOrHidePlotTab(plot_panel, False, True)

        show_pca = plot_data_fetcher.isPcaDataKey(key)
        for plot_panel in self.__plot_panels:
            visible = self.__central_tab.indexOf(plot_panel) > -1

            if plot_data_fetcher.isSummaryKey(key):
                show_plot = plot_panel.supportsPlotProperties(time=True, value=True, histogram=True, pca=show_pca)
                self.showOrHidePlotTab(plot_panel, visible, show_plot)

            elif plot_data_fetcher.isBlockObservationKey(key):
                show_plot = plot_panel.supportsPlotProperties(depth=True, value=True, pca=show_pca)
                self.showOrHidePlotTab(plot_panel, visible, show_plot)

            elif plot_data_fetcher.isGenKWKey(key):
                show_plot = plot_panel.supportsPlotProperties(value=True, histogram=True, pca=show_pca)
                self.showOrHidePlotTab(plot_panel, visible, show_plot)

            elif plot_data_fetcher.isGenDataKey(key):
                show_plot = plot_panel.supportsPlotProperties(index=True, pca=show_pca)
                self.showOrHidePlotTab(plot_panel, visible, show_plot)

            elif plot_data_fetcher.isPcaDataKey(key):
                show_plot = plot_panel.supportsPlotProperties(pca=show_pca)
                self.showOrHidePlotTab(plot_panel, visible, show_plot)

            else:
                raise NotImplementedError("Key %s not supported." % key)

        self.__plot_panel_tracker.restorePlotType(plot_data_fetcher, key)

        self.__central_tab.blockSignals(False)
        self.currentPlotChanged()

        if self.checkPlotStatus():
            self.plotSettingsChanged()
