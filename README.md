<div align="center"><h2> Wide Open Gazes: Quantifying Visual Exploratory Behavior in Soccer with Pose Enhanced Positional Data </h2></div>

### Introduction
Traditional approaches to measuring visual exploratory behavior in soccer rely on counting visual exploratory actions (VEAs) based on rapid head movements. These methods suffer from player position bias (i.e., a focus on central midfielders), annotation challenges, binary measurement constraints, lack the power to measure relevant in-game success, and are incompatible with fundamental soccer analytics models such as pitch control.

This research introduces a novel formulaic continuous stochastic vision layer to quantify players’ visual perception from pose-enhanced spatiotemporal tracking that includes head and shoulder rotations.

---

### Methods
We build a comprehensive two-dimensional, top-down representation of a player’s visual perception, by creating two complementary models: a **field of view model** to quantify a player’s ability to see other players on the pitch, and an **occlusion model** to describe any player on the pitch that blocks another player’s vision. Combined, these two facets give us a complete **vision map** of a player, as shown in Figure 1.
 
<div align="center">
  <img src="assets/vision.gif" alt="SHAP Value Image" width="600">
  <p><strong>Figure 1.</strong> The vision map of Argentina's #7 (Rodrigo De Paul)</p>
</div>

We quantify this **field of view** through a probabilistic model that utilizes a player's speed and head orientation, modeling human vision where objects in the peripheral vision or at greater distances are more difficult to locate exactly, especially at high speeds.

We further enhance this field of view map by incorporating visual obstructions caused by other players on the field. Within this **occlusion map**, the size of occlusion caused by each player <i>j</i> is influenced by the distance to player <i>i</i> and by the shoulder rotation of player <i>j</i>. If player <i>j</i> is angled sideways from the perspective of player <i>i</i>, they occupy less space in their field of vision compared to when looking directly at their torso.
After merging these two maps into **vision maps**, we further combine them with pitch control and pitch value surfaces to analyze a player's exploratory behavior while awaiting passes and how this influences the outcome of subsequent dribbles after receiving these passes.
 

<div align="center">
  <img src="assets/shap.png" alt="SHAP Value Image" width="600">
  <p><strong>Figure 2.</strong> The impact and magnitude of individual variables of the XGBoost Classifier that includes all features, using SHAP values. Aggregated visual metrics labeled A – H.</p>
</div>

---

### Results
Through an ablation study, that begins with a simple baseline (distance to goal) and that progressively adds more complex features, we demonstrate that aggregated visual metrics captured during moments while awaiting a pass (listed in Figure 2) predict significant pitch value gained or lost following subsequent dribbles.

---

### Conclusion
This research introduces a novel method for modeling visual perceptual behavior in sports as a two-dimensional top-down plane. It integrates pitch control and pitch value with vision maps to quantify controlled and observed space. We show that players who observe higher quantities of occupied space while awaiting passes exhibit larger gains in their spatial positioning after subsequent on-ball actions. Unlike traditional VEA counts, this method is position independent, requires no manual annotation, and offers continuous measurements compatible with existing analytics models.


