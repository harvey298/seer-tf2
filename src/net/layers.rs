use tch::nn;

use crate::{MOVEMENT_INPUT_SIZE, ANGLE_INPUT_SIZE};

#[derive(Debug)]
pub struct PlayerMovementDetector {
    pub layer1: nn::Linear,
    pub layer2: nn::Linear,
    pub layer3: nn::Linear,
    pub layer4: nn::Linear,
    pub layer5: nn::Linear,
}

impl PlayerMovementDetector {
    pub fn init(vs: &nn::Path) -> Self {

        let layer1 = nn::linear(vs, MOVEMENT_INPUT_SIZE, MOVEMENT_INPUT_SIZE/2, Default::default());
        let layer2 = nn::linear(vs, MOVEMENT_INPUT_SIZE/2, MOVEMENT_INPUT_SIZE/4, Default::default());
        let layer3 = nn::linear(vs, MOVEMENT_INPUT_SIZE/4, 64, Default::default());
        let layer4 = nn::linear(vs, 64, 12, Default::default());
        let layer5 = nn::linear(vs, 12, 1, Default::default());

        Self { layer1, layer2, layer3, layer4, layer5 }
    }
}


#[derive(Debug)]
pub struct PlayerLookDetector {
    pub layer1: nn::Linear,
    pub layer2: nn::Linear,
    pub layer3: nn::Linear,
    pub layer4: nn::Linear,
    pub layer5: nn::Linear,
}

impl PlayerLookDetector {
    pub fn init(vs: &nn::Path) -> Self {

        let layer1 = nn::linear(vs, ANGLE_INPUT_SIZE, ANGLE_INPUT_SIZE/2, Default::default());
        let layer2 = nn::linear(vs, ANGLE_INPUT_SIZE/2, ANGLE_INPUT_SIZE/4, Default::default());
        let layer3 = nn::linear(vs, ANGLE_INPUT_SIZE/4, 64, Default::default());
        let layer4 = nn::linear(vs, 64, 12, Default::default());
        let layer5 = nn::linear(vs, 12, 1, Default::default());

        Self { layer1, layer2, layer3, layer4, layer5 }
    }
}