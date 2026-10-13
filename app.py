import streamlit as st
from PIL import Image
import pandas as pd

from predict import predict_image

st.set_page_config(page_title="Nutrient Deficiency Detection", page_icon="🌽", layout="centered")

RECOMMENDATIONS = {
    "Healthy": {
        "explanation": "The model detected visual characteristics associated with a healthy plant.",
        "guidance": [
            "Continue appropriate crop management.",
            "Monitor plant health regularly.",
            "Maintain suitable soil and irrigation practices."
        ]
    },
    "Nitrogen": {
        "explanation": "The model detected visual characteristics associated with nitrogen deficiency.",
        "guidance": [
            "Verify nitrogen status through appropriate soil/plant assessment.",
            "Review crop nutrient management.",
            "Apply fertilizer only according to agronomic recommendations and local conditions."
        ]
    },
    "Phosphorus": {
        "explanation": "The model detected visual characteristics associated with phosphorus deficiency.",
        "guidance": [
            "Verify phosphorus status through appropriate soil/plant assessment.",
            "Review soil fertility and nutrient management.",
            "Follow locally appropriate agronomic recommendations."
        ]
    },
    "Potassium": {
        "explanation": "The model detected visual characteristics associated with potassium deficiency.",
        "guidance": [
            "Verify potassium status through appropriate soil/plant assessment.",
            "Review crop nutrient management.",
            "Follow locally appropriate agronomic recommendations."
        ]
    },
    "Magnesium": {
        "explanation": "The model detected visual characteristics associated with magnesium deficiency.",
        "guidance": [
            "Verify magnesium status through appropriate soil/plant assessment.",
            "Review soil fertility and nutrient management.",
            "Follow locally appropriate agronomic recommendations."
        ]
    }
}

def main():
    st.title("AI-Based Plant Nutrient Deficiency Detection")
    st.write("AI-powered analysis of maize leaf images for visual nutrient deficiency classification.")
    
    st.write("---")

    st.subheader("Upload Maize Leaf Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Leaf"):
                with st.spinner("Analyzing image..."):
                    # Process prediction
                    results = predict_image(image)
                    
                    predicted_class = results["predicted_class"]
                    confidence = results["confidence"]
                    probs = results["probabilities"]
                    
                    st.write("---")
                    st.subheader("Analysis Result")
                    
                    st.markdown(f"**Predicted Condition:**<br>{predicted_class}", unsafe_allow_html=True)
                    st.markdown(f"**Model Confidence:**<br>{confidence:.2%}", unsafe_allow_html=True)
                    
                    st.write("---")
                    st.subheader("Probability Distribution")
                    
                    # Prepare data for chart
                    prob_df = pd.DataFrame(
                        list(probs.items()), 
                        columns=["Condition", "Probability"]
                    )
                    
                    # Display as a bar chart
                    st.bar_chart(prob_df.set_index("Condition"))
                    
                    st.write("---")
                    st.subheader("General Guidance")
                    
                    rec = RECOMMENDATIONS.get(predicted_class)
                    if rec:
                        st.write(rec["explanation"])
                        st.write("**Recommended next steps:**")
                        for step in rec["guidance"]:
                            st.markdown(f"• {step}")
                            
        except Exception as e:
            st.error("An error occurred during processing. Please ensure the image is valid.")
            st.write(f"Error details: {str(e)}")

    st.write("---")
    st.info(
        "**Important:**\n"
        "This system provides an AI-based prediction from leaf appearance. "
        "It is intended for informational purposes and should not replace "
        "appropriate agricultural, soil, or expert assessment."
    )

    st.write("---")
    
    with st.expander("About the Model"):
        st.markdown("""
        **Model:** EfficientNet-B0 with transfer learning  
        **Task:** 5-class maize leaf classification  
        **Classes:** Healthy, Magnesium, Nitrogen, Phosphorus, Potassium  

        **Dataset Splits:**
        * Training images: 324
        * Validation images: 69
        * Held-out test images: 70
        
        **Official Test Performance:**
        * Official test accuracy: 84.29%
        * Macro F1: 0.8248
        
        *(Official test performance on the project's held-out test set.)*
        """)

    with st.expander("Model Evaluation"):
        st.markdown("""
        **Internal model experiments:**

        * **Experiment 1 — Frozen:**
          * 80.00% accuracy
          * Macro F1 = 0.7482

        * **Experiment 2 — Fine-tuned:**
          * 82.86% accuracy
          * Macro F1 = 0.7747

        * **Experiment 3 — Class-weighted:**
          * 81.43% accuracy
          * Macro F1 = 0.7576

        * **Experiment 4 — Weighted sampling:**
          * 82.86% accuracy
          * Macro F1 = 0.7739

        * **Experiment 5 — Deeper fine-tuning (Final Model):**
          * 84.29% accuracy
          * Macro F1 = 0.8248
        """)

    with st.expander("Limitations"):
        st.markdown("""
        * The current model is trained exclusively for maize leaves.
        * The total dataset contains 463 images.
        * The dataset has class imbalance.
        * The held-out test set contains 70 images.
        * Potassium has only 8 samples in the held-out test set.
        * Real-world field performance has not been established.
        * External validation has not yet been performed.
        * Nutrient status should always be confirmed through appropriate agricultural or soil assessment.
        """)

if __name__ == "__main__":
    main()
