import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import load

global modelRF

def changeAttrition(option):
    if option == "No":
        return 0
    elif option == "Yes":
        return 1
def changeTravel(trav):
    if trav == "Travel_Rarely":
        return 0
    elif trav == "Travel_Frequently":
        return 1
    elif trav == "Non-Travel":
        return 2
def changeDep(department):
    if department == "Sales":
        return 0
    elif department == "Research & Development":
        return 1
    elif department == "Human Resources":
        return 2
def changestudyField(field):
    if field == "Life Sciences":
        return 0
    elif field == "Other":
        return 1
    elif field == "Medical":
        return 2
    elif field == "Marketing":
        return 3
    elif field == "Technical Degree":
        return 4
    elif field == "Human Resources":
        return 5
def changeSex(sex):
    if sex == "Female":
        return 0
    elif sex == "Male":
        return 1
def changeFunction(func):
    if func == "Healthcare Representative":
        return 0
    elif func == "Research Scientist":
        return 1
    elif func == "Sales Executive":
        return 2
    elif func == "Marketing":
        return 3
    elif func == "Human Resources":
        return 4
    elif func == "Research Director":
        return 5
    elif func == "Laboratory Technician":
        return 6
    elif func == "Manufacturing Director":
        return 7
    elif func == "Sales Representative":
        return 8
    elif func == "Manager":
        return 9
def changeMarStatus(status):
    if status == "Married":
        return 0
    elif status == "Single":
        return 1
    elif status == "Divorced":
        return 2
def carregarDFS():

    dfEmployee = pd.read_csv("G:/Meu Drive/Trabalho/Portfolio/HRProject/data/employee_survey_data.csv")
    dfGeneral = pd.read_csv("G:/Meu Drive/Trabalho/Portfolio/HRProject/data/general_data.csv")
    dfManagerData = pd.read_csv("G:/Meu Drive/Trabalho/Portfolio/HRProject/data/manager_survey_data.csv")

    df = pd.merge(dfEmployee, dfGeneral, on="EmployeeID", how="inner")
    dfTotal = pd.merge(df, dfManagerData, on="EmployeeID", how="inner")
    dfTotal.dropna(axis=0, inplace=True)
    dfTotal.drop("EmployeeID", axis=1, inplace=True)
    dfTotal.drop(["Over18", "JobInvolvement", "EmployeeCount", "StandardHours"], axis=1, inplace=True)

    dfCat = dfTotal.select_dtypes("object")
    dfCatChange = dfCat.copy()
    dfCatChange["AttritionNum"] = dfCat.Attrition.map(changeAttrition)
    dfCatChange["TravelNum"] = dfCat.BusinessTravel.map(changeTravel)
    dfCatChange["DepartmentNum"] = dfCat.Department.map(changeDep)
    dfCatChange["EduFieldNum"] = dfCat.EducationField.map(changestudyField)
    dfCatChange["genderNum"] = dfCat.Gender.map(changeSex)
    dfCatChange["jobRoleNum"] = dfCat.JobRole.map(changeFunction)
    dfCatChange['maritalStatusNum'] = dfCat.MaritalStatus.map(changeMarStatus)
    dfCatChange.drop(dfCat.columns.values, axis=1, inplace=True)

    dfFinal = pd.concat([dfCatChange, dfTotal.drop(dfCat.columns.values, axis=1)], axis=1)
    return dfFinal

def realizaPrevisao(dados):
    """
    Esta funcao realiza a predicao recebendo para tal um modelo e um dataframe em formato JSON
    A função:
    1. converte o JSON que deverá ser enviado em um formato específico
    2. Passa o valor convertido para o modelo que realiza as previsões
    3. retorna as previsões

    :param modelo:
    :param dados:
    :return:
    """
    try:
        global modelRF
        previsoes = modelRF.predict(dados)
        return previsoes
    except:
        raise("Erro ao realizar previsoes")

def loadModel(modelPath):
    try:
        global modelRF
        modelRF = load(modelPath)
    except:
        raise("Erro ao carregar o modelo")
if __name__ == "__main__":

    modelPath = '../modelRFHR.joblib'
    global modelRF
    modelRF = load(modelPath)