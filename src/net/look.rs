use tch::{nn::{VarStore, self, OptimizerConfig, Module}, Device, Scalar, Kind, Tensor, Reduction};
use anyhow::Result;

use super::layers::PlayerLookDetector;

#[derive(Debug)]
pub struct LookDetector {
    layers: PlayerLookDetector,
    var_store: VarStore,
    optimizer: nn::Optimizer,
    device: Device,
}

impl LookDetector {
    pub fn new(var_store: Option<VarStore>) -> Result<Self> {
        let device = Device::cuda_if_available();

        let var_store = var_store.unwrap_or(VarStore::new(device));

        let vs = &var_store.root();

        let layers = PlayerLookDetector::init(vs);

        let opt_config = nn::Adam::default().amsgrad(true);

        let opt: nn::Optimizer = opt_config.build(&var_store, 1e-4)?;

        let me = Self { layers, var_store, optimizer: opt, device };

        Ok(me)
    }

    pub fn train(&mut self, training_data: &Tensor, training_labels: &Tensor, test_data: &Tensor, test_labels: &Tensor, epochs: i32) -> Result<f32> {

        let training_data = training_data.to_dense(Kind::Float);
        let training_labels = training_labels.to_dense(Kind::Float);
        let test_data = test_data.to_dense(Kind::Float);
        // let test_labels = test_labels.to_dense(Kind::Float);

        for epoch in 1..=epochs {

            let loss: Tensor = self.forward(&training_data);


            let loss: Tensor = loss.binary_cross_entropy::<Tensor>(&training_labels, None, Reduction::Mean);

            self.optimizer.backward_step(&loss);

            if epoch % 100 == 0 {

                let test_output = self.forward(&test_data).sigmoid();

                // TODO: Fix this - it doesn't work as intended, its for bikeshedding
                let accuracy = test_output.eq(Scalar::float(1.0)).mean(Kind::Float).double_value(&[]);

                println!("Epoch: {}, Test Accuracy: {:.2}", epoch, accuracy);
            }
        }

        Ok(0.0)
    }

    pub fn borrow_var_store(&mut self) -> &mut VarStore { &mut self.var_store }
}

impl Module for LookDetector {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        let xs = xs.to_kind(self.layers.layer1.ws.kind());

        let x = self.layers.layer1.forward(&xs).relu();
        let x = self.layers.layer2.forward(&x).relu();
        let x = self.layers.layer3.forward(&x).relu();
        let x = self.layers.layer4.forward(&x).relu();
        let x = self.layers.layer5.forward(&x).relu();
        let x = x.sigmoid();

        x
    }
}