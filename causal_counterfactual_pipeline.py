import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from dowhy import CausalModel
from preprocessing import preprocess_shot_data
from causal_utils import (
    prepare_data,
    build_causal_model,
    estimate_effect,
    get_boundary_cases,
    generate_counterfactuals,
    evaluate_counterfactual_quality,
    run_model_pipeline,
    evaluate_counterfactual_proportions,
    plot_results
)

warnings.filterwarnings("ignore")
plt.style.use("ggplot")
np.random.seed(42)

if __name__ == "__main__":
    print("=== Loading and preprocessing data ===")
    df = preprocess_shot_data(start_season=2016, final_season=2023)
    df = prepare_data(df)

    treatment_var = 'emptyNetHome'

    print("\n=== Causal Analysis ===")
    causal_model = build_causal_model(df, treatment_var, 'result')
    if causal_model:
        estimate = estimate_effect(causal_model)
        effect_size = min(max(estimate.value, 0.1), 0.9) if estimate else 0.3
    else:
        effect_size = 0.3

    print("\n=== Model Evaluation (Original) ===")
    orig_metrics, orig_model, test_data, preprocessor, model_for_filtering = run_model_pipeline(
        df, model_name="Original"
    )

    print("\n=== Extract Model-Boundary Positives ===")
    df_boundary = get_boundary_cases(
        df, model_for_filtering, preprocessor,
        numerical_features=['distance_to_net', 'shot_angle', 'period', 'powerplayHome', 
                           'powerplayAway', 'emptyNetHome', 'emptyNetAway'],
        categorical_features=['shotType', 'goalie', 'HomevsAway']
    )

    print("\n=== Counterfactual Generation ===")
    cf1 = generate_counterfactuals(
        df_boundary, 'emptyNetHome', effect_size, 0.2, model_for_filtering, preprocessor,
        numerical_features=['distance_to_net', 'shot_angle', 'period', 'powerplayHome', 
                           'powerplayAway', 'emptyNetHome', 'emptyNetAway'],
        categorical_features=['shotType', 'goalie', 'HomevsAway']
    )
    cf2 = generate_counterfactuals(
        df_boundary, 'powerplayHome', effect_size, 0.2, model_for_filtering, preprocessor,
        numerical_features=['distance_to_net', 'shot_angle', 'period', 'powerplayHome', 
                           'powerplayAway', 'emptyNetHome', 'emptyNetAway'],
        categorical_features=['shotType', 'goalie', 'HomevsAway']
    )
    cf3 = generate_counterfactuals(
        df_boundary, 'shotType', effect_size, 0.2, model_for_filtering, preprocessor,
        numerical_features=['distance_to_net', 'shot_angle', 'period', 'powerplayHome', 
                           'powerplayAway', 'emptyNetHome', 'emptyNetAway'],
        categorical_features=['shotType', 'goalie', 'HomevsAway']
    )
    counterfactuals = pd.concat([cf1, cf2, cf3], ignore_index=True)
    print(f"Generated {len(counterfactuals)} combined counterfactual samples")

    print("\n=== Counterfactual Quality Evaluation ===")
    if len(counterfactuals) > 0:
        quality_results = evaluate_counterfactual_quality(
            df, counterfactuals, features=['distance_to_net', 'shot_angle']
        )
    else:
        print("No counterfactuals to evaluate")

    print("\n=== Counterfactual Proportion Analysis ===")
    if len(counterfactuals) > 0:
        proportion_metrics, proportion_models, test_data = evaluate_counterfactual_proportions(
            df, counterfactuals, proportions=[0.05, 0.1, 0.5, 1.0]
        )
        print(proportion_metrics.to_markdown())
    else:
        proportion_metrics, proportion_models, test_data = None, None, None
        print("Skipping proportion analysis - no counterfactuals generated")

    print("\n=== Results Summary ===")
    results = pd.DataFrame([orig_metrics] + (proportion_metrics.to_dict('records') if proportion_metrics is not None else []))
    print(results.to_markdown())

    print("\n=== Plotting ROC and Precision-Recall Curves ===")
    if proportion_metrics is not None:
        plot_results(orig_metrics, orig_model, proportion_metrics.to_dict('records'), proportion_models, test_data)
    else:
        print("Not enough data for comparison plots")
