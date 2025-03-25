from model.randomforest import RandomForest
from model.chained_model import ChainedModel
from model.hierarchical_model import HierarchicalModel
from Config import Config


def model_predict(data, df, name):
    if Config.CLASSIFICATION_APPROACH == Config.CHAINED_OUTPUT:
        # Chained Multi-Output Classification (Decision 1)
        print("=== Running Chained Multi-Output Classification ===")
        model = ChainedModel("ChainedModel", data.get_embeddings(), data.y_train_dict)
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)
    elif Config.CLASSIFICATION_APPROACH == Config.HIERARCHICAL:
        # Hierarchical Modeling (Decision 2)
        print("=== Running Hierarchical Modeling ===")
        model = HierarchicalModel("HierarchicalModel", data.get_embeddings(), data.y_train_dict)
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)
    else:
        # Original single-label classification (for backward compatibility)
        print("=== Running Single-Label Classification (RandomForest) ===")
        model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)