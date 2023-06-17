use std::{fs, collections::HashMap, path::Path};

use anyhow::Result;
use serde::{Serialize, Deserialize};
use tch::{Tensor, nn::{Module, VarStore}};
use tf_demo_parser::{Demo, DemoParser, demo::{parser::gamestateanalyser::GameStateAnalyser, vector::Vector}};

use crate::{ANGLE_INPUT_SIZE, MOVEMENT_INPUT_SIZE};

mod layers;
pub mod movement;
pub mod look;

pub const LOOK_SAVE_PATH: &str = "./data/look.safetensors";
pub const MOVEMENT_SAVE_PATH: &str = "./data/movement.safetensors";

pub struct Seer {
    look: look::LookDetector,
    movement: movement::MovementDetector,
}

impl Seer {
    pub fn new() -> Result<Self> {
        let mut look = look::LookDetector::new(None)?;
        let mut movement = movement::MovementDetector::new(None)?;

        if !Path::new("./data/").exists() { fs::create_dir_all("./data/")?; }

        if Path::new(LOOK_SAVE_PATH).exists() { look.borrow_var_store().load(LOOK_SAVE_PATH)?; }

        if Path::new(MOVEMENT_SAVE_PATH).exists() { movement.borrow_var_store().load(MOVEMENT_SAVE_PATH)?; }

        Ok(Self { look: look, movement: movement })
    }

    pub fn train(&mut self, training_data: &[TrainingData], save: bool) -> Result<()> {

        for data in training_data {
            
            // Open the demo file
            let tensors = demo_to_tensor(&data.demo_file, None);

            for tensor in tensors {
                let mut cheating = data.cheater_steam_id.clone().unwrap_or(Vec::new()).contains(&tensor.steam_id);
                let label = Tensor::from_slice(&[if cheating { 1.0 } else { 0.0 }]);
                
                // TODO: Make test data and labels actualy work
                self.look.train(&tensor.look_tensor, &label, &label, &label, 32)?;
                self.movement.train(&tensor.position_tensor, &label, &label, &label, 32)?;

            }           

        }

        if save { self.save()? }
        
        Ok(())
    }

    /// TODO
    pub fn save(&mut self) -> Result<()> {

        self.look.borrow_var_store().save(LOOK_SAVE_PATH)?;
        self.movement.borrow_var_store().save(MOVEMENT_SAVE_PATH)?;        

        Ok(())
    }

    pub fn scan_demo(&self, path: &str) -> Result<()> {

        let data = demo_to_tensor(path, None);

        for player in &data {

            let movement_result = tensor_to_vec(self.movement.forward(&player.position_tensor));
            let look_result = tensor_to_vec(self.look.forward(&player.look_tensor));

            let movement_result = movement_result.first().unwrap();
            let look_result = look_result.first().unwrap();

            let id = &player.steam_id;

            // TODO: have this outputted in another way
            println!("{id}: {movement_result} | {look_result}");
        }


        Ok(())
    }
}

fn open_demo(path: &str) -> Result<HashMap<String, Vec<Player>>> {
    let file = fs::read(path).unwrap();

    let demo = Demo::new(&file);
    let parser  = DemoParser::new_all_with_analyser(demo.get_stream(), GameStateAnalyser::default());
    let (_, mut state) = parser.ticker().unwrap();

    let mut player_buffer: HashMap<String, Vec<Player>> = HashMap::new();

    loop {
        match state.tick() {
            Ok(true) => {

                let state2 = state.state();
                
                for player in &state2.players {
                    if let Some(info) = &player.info {
                        let steam_id= info.steam_id.clone();
                        
                        let pitch_angle = player.pitch_angle;
                        let position = player.position;
                        let view_angle = player.view_angle;
            
                        let player = Player {
                            steam_id: steam_id.clone(),
                            position,
                            view_angle,
                            pitch_angle,
                        };
            
                        if player_buffer.contains_key(&steam_id) {
                            player_buffer.get_mut(&steam_id).unwrap().push(player);
                        } else {
                            player_buffer.insert(steam_id, vec![player]);
                        }
                    }
                }                
                
                continue;
            }
            Ok(false) => {
                break;
            }
            Err(e) => {
                println!("Error: {e:?}");
                break;
            }
        }
    }

    Ok(player_buffer)
}

/// TODO: rename variables to actually have meaning
/// TODO: Make this return a Result
/// if player is None it will turn every player into their own tensor, if player is not None it turn the selected player into a tensor
/// Player would be a steam id
fn demo_to_tensor(path: &str, players: Option<&[String]>) -> Vec<PlayerDemoTensor> {
    let mut player_buffer = open_demo(path).unwrap();

    let player_buffer = if players.is_some() {
        let mut buffer = HashMap::new();
        for player in players.unwrap() {
            buffer.insert(player.to_owned() , player_buffer.remove(player).unwrap());
        }
        buffer
    } else { player_buffer };

    let mut results = Vec::new();

    // convert to tensor
    for player in player_buffer.into_iter() {
        let steam_id = player.0;
        let player_info = player.1;
        let mut c1: Vec<f64> = Vec::new(); 
        let mut c2: Vec<f64> = Vec::new(); 

        let mut c3: Vec<Vec<f64>> = Vec::new();

        for (_, player) in player_info.into_iter().enumerate() {
            if !( (c1.len() + c2.len())  >= ANGLE_INPUT_SIZE as usize) {
                c1.push(player.view_angle as f64);
                c2.push(player.pitch_angle as f64);
            }

            let position = player.position;

            let x = position.x as f64;
            let y = position.y as f64;
            let z = position.z as f64;
            let pos =  vec![x, y, z];
            
            c3.push(pos);
            
        }

        let mut movement_buffer = Vec::new();
        for c in &c3 {
            for item in c {
                if !(movement_buffer.len() >= MOVEMENT_INPUT_SIZE as usize) {
                    movement_buffer.push(item.clone());
                }
            }
        }

        let mut look_buffer = c1;
        look_buffer.append(&mut c2);

        // println!("Size: {} | {}", movement_buffer.len(), look_buffer.len());

        while !(look_buffer.len() >= ANGLE_INPUT_SIZE as usize) { look_buffer.push(0.0); }

        while !(movement_buffer.len() >= MOVEMENT_INPUT_SIZE as usize) { movement_buffer.push(0.0); }

        let look_tensor = Tensor::from_slice(&look_buffer).reshape(&[1, ANGLE_INPUT_SIZE]);
        let position_tensor = Tensor::from_slice(&movement_buffer).reshape(&[1, MOVEMENT_INPUT_SIZE]);

        results.push(PlayerDemoTensor {
            steam_id,
            look_tensor,
            position_tensor,
        });
    }
    
    results
}

#[derive(Debug)]
/// Represents a Player as a tensor over the entire game length
pub struct PlayerDemoTensor {
    pub steam_id: String,
    pub position_tensor: Tensor,
    pub look_tensor: Tensor
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingData {
    /// Path to the demo file
    pub demo_file: String,
    pub cheater: bool,
    /// To be left as None if no cheater is present!
    pub cheater_steam_id: Option<Vec<String>>,

    /// Whether or not to exclude the demo file in the training data
    pub exclude: Option<bool>,
}

#[derive(Debug, Default)]
pub struct Player {
    pub steam_id: String,
    pub position: Vector,
    pub view_angle: f32,
    pub pitch_angle: f32,
}

pub fn tensor_to_vec(tensor: Tensor) -> Vec<f64> {
    let mut itr = tensor.clone(&tensor).view(-1).iter::<f64>().unwrap();
    
    let mut data: Vec<f64> = Vec::new();

    loop {
        if let Some(item) = itr.next() {
            data.push(item.clone())
        } else {
            break
        }
    }

    data
}