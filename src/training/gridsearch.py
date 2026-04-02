import pickle as pkl
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from src.data.agnews import AGNews
from datetime import datetime
from src.const import DEBUG, MODEL_DIR, RANDOM_SEED, LOGGER
from src.utils.output import get_output_path
from src.training.eval import evaluate_model


def svm_gridsearch(
    ds: AGNews,
    param_grid={},
    eval: bool = True,
    assignment: int = 1,
) -> None:
    """Perform grid search for SVM hyperparameters and evaluate the best model.

    Args:
        ds (AGNews): The AGNews dataset object containing training and
                     evaluation data.
        param_grid (dict, optional): The parameter grid for grid search.
                                     Defaults to {}.
        eval (bool, optional): Whether to evaluate the best model. Defaults to
                               True.
        assignment (int, optional): The assignment number for output path.
                                    Defaults to 1.
    """
    # Grid search with k-fold cross-validation
    LOGGER.info(f"Starting SVM Grid Search with parameters: {param_grid}")
    svm_model = SVC(kernel="Linear", max_iter=10000, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        refit=True,
        scoring="accuracy",
        verbose=2 if DEBUG else 0,
    )

    grid_search.fit(ds.X_train, ds.y_train)
    LOGGER.info(
        f"Grid search completed. Best score: "
        f"{grid_search.best_score_:.4f} with params: "
        f"{grid_search.best_params_}"
    )

    # Get output directory for this assignment
    output_dir = get_output_path(assignment=assignment)
    out_path = (
        output_dir / f"svm_gridsearch_results_{datetime.now().isoformat()}.csv"
    )

    results = grid_search.cv_results_
    with open(out_path, "w") as f:
        f.write("params,mean_test_score,std_test_score\n")
        for i in range(len(results["params"])):
            params = results["params"][i]
            mean_score = results["mean_test_score"][i]
            std_score = results["std_test_score"][i]
            f.write(f"{params},{mean_score:.4f},{std_score:.4f}\n")

    LOGGER.info(f"Grid search results saved to {out_path}")

    model_path = MODEL_DIR / "svm_model.pkl"
    with open(model_path, "wb") as f:
        pkl.dump(grid_search.best_estimator_, f)
    LOGGER.info(f"Best model saved to {model_path}")

    if eval:
        evaluate_model(grid_search.best_estimator_, ds)
