from ert.enkf.ert_workflow_list_handler import ErtWorkflowListHandler
from ert_gui.models import ErtConnector
from ert_gui.models.mixins import ChoiceModelMixin
from ert_gui.models.mixins.list_model import ListModelMixin


class WorkflowsModel(ErtConnector, ListModelMixin, ChoiceModelMixin):

    def __init__(self):
        self.__value = None
        super(WorkflowsModel, self).__init__()

    def getList(self):
        return sorted(self.ert().getWorkflowList().getWorkflowNames(), key=lambda v: v.lower())

    def getChoices(self):
        return self.getList()

    def getCurrentChoice(self):
        if self.__value is None:
            return self.getList()[0]
        return self.__value

    def setCurrentChoice(self, value):
        self.__value = value
        self.observable().notify(self.CURRENT_CHOICE_CHANGED_EVENT)


    def createWorkflowRunner(self):
        workflow_name = self.getCurrentChoice()
        workflow_list = self.ert().getWorkflowList()
        enkf_main_pointer = workflow_list.parent().from_param(workflow_list.parent())
        return ErtWorkflowListHandler(workflow_list, workflow_name, enkf_main_pointer)


    def getError(self):
        error = self.ert().getWorkflowList().getLastError()

        error_message = ""

        for error_line in error:
            error_message += error_line + "\n"

        return error_message
