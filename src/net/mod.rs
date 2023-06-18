use std::{fs, collections::{HashMap, VecDeque}, path::Path, time::Instant, rc::Rc, thread::spawn};

use anyhow::Result;
use crossbeam_channel::unbounded;
use crypto_hash::{hex_digest, Algorithm};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde::{Serialize, Deserialize};
use tch::{Tensor, nn::{Module, VarStore}};
use tf_demo_parser::{Demo, DemoParser, demo::{parser::gamestateanalyser::GameStateAnalyser, vector::Vector}};
use uuid::Uuid;

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
        println!("Training...");

        for data in training_data {

            let demo_file_path = &data.demo_file;
            
            // Open the demo file
            let tensors: Vec<PlayerDemoTensor> = demo_to_tensor(&demo_file_path, None);
            let batches = tensors.len();

            let path = data.demo_file.replace("\\", "/");
            let path = path.replace(".dem", ".replay");

            if !Path::new(&path).exists() {

                let steam_ids = data.clone().cheater_steam_id.unwrap_or(Vec::new());

                let demo_file_path = demo_file_path.clone();
                
                spawn(move || {
                    let data: Vec<PlayerDemoTensor> = demo_to_tensor(&demo_file_path, None);
                    let steam_ids = steam_ids.clone();

                    let mut buffer = Vec::new();

                    for tensor in data.clone() {
                        let cheating = steam_ids.contains(&tensor.steam_id);

                        let mut new_tensor = tensor.clone();
                        new_tensor.cheating = cheating;
                        new_tensor.steam_id = Uuid::new_v4().to_string();

                        buffer.push(new_tensor);
                    }

                    save_replay(buffer, &path, &steam_ids).unwrap();
                });
    
            }

            for (index, tensor) in tensors.into_iter().enumerate() {
                
                println!("Training on a batch {index}/{batches}...");

                let cheating = data.cheater_steam_id.clone().unwrap_or(Vec::new()).contains(&tensor.steam_id);
                let label = Tensor::from_slice(&[if cheating { 1.0 } else { 0.0 }]);

                
                // TODO: Make test data and labels actualy work
                for look_tensor in tensor.look_tensors {
                    self.look.train(&look_tensor, &label, &label, &label, 32)?;    
                }
                
                for position_tensor in tensor.position_tensors {
                    self.movement.train(&position_tensor, &label, &label, &label, 32)?;
                }

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

    /// Scans a demo
    /// Will convert the demo into a Seer replay (a json file)
    pub fn scan_demo(&self, path: &str) -> Result<()> {

        let path = if Path::new(&path.replace(".dem", ".replay")).exists() {
            path.replace(".dem", ".replay")
        } else { path.to_string() };

        let data: Vec<PlayerDemoTensor> = demo_to_tensor(&path, None);

        for player in &data {

            let mut look_results = Vec::new();
            for look_tensor in &player.look_tensors {
                let look_result = tensor_to_vec(self.look.forward(&look_tensor));
                look_results.push(look_result.first().unwrap().clone());
            }

            let mut movement_results = Vec::new();
            for position_tensor in &player.position_tensors {
                // println!("Working on batch");
                let movement_result = tensor_to_vec(self.movement.forward(&position_tensor));
                movement_results.push(movement_result.first().unwrap().clone());
            }

            let movement_results_size = movement_results.len();
            let movement_result: f64 = movement_results.into_iter().sum::<f64>()/movement_results_size as f64;

            let look_results_size = look_results.len();
            let look_result: f64 = look_results.into_iter().sum::<f64>()/look_results_size as f64;

            let id = &player.steam_id;

            // TODO: have this outputted in another way
            println!("{id}: {movement_result} | {look_result}");
        }

        let path = path.replace(".dem", ".replay");
        if !Path::new(&path).exists() {

            save_replay(data, &path, &[])?;

        }
        
        Ok(())
    }


}

fn open_demo(path: &str) -> Result<HashMap<String, Vec<Player>>> {

    let file = fs::read(path).unwrap();

    let demo = Demo::new(&file);
    let parser  = DemoParser::new_all_with_analyser(demo.get_stream(), GameStateAnalyser::default());
    let (header, mut state) = parser.ticker().unwrap();

    let tick_trate = header.frames as f32/header.duration;

    println!("{tick_trate}");

    let mut player_buffer: HashMap<String, Vec<Player>> = HashMap::new();    

    loop {
        match state.tick() {
            Ok(true) => {

                let state2 = state.state();
                
                for player in &state2.players {
                    if let Some(info) = &player.info {
                        let steam_id= info.steam_id.clone();

                        // println!("{}",steam_id);
                        
                        let pitch_angle = player.pitch_angle;
                        let position = player.position;
                        let view_angle = player.view_angle;

                        // println!("{steam_id}s Position: {:?}",position);
            
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
/// This will convert the demo file into a one quickly usable by Seer (will maintain the originial)
fn demo_to_tensor(path: &str, players: Option<&[String]>) -> Vec<PlayerDemoTensor> {

    let rply_path = path.replace("\\", "/");
    let rply_path = rply_path.replace(".dem", ".replay");

    // Assume that its saved as a PlayerDemoTensorSaveAble
    if path.contains(".replay") || Path::new(&rply_path).exists() {
        let data: SeerDemo = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();

        let out = data.data.into_iter().map(|f | { PlayerDemoTensor::from_saveable(f) }).collect();

        return out;
    }

    let mut player_buffer = open_demo(path).unwrap();
    let work_start = Instant::now();

    let player_buffer = if players.is_some() {
        let mut buffer = HashMap::new();
        for player in players.unwrap() {
            buffer.insert(player.to_owned() , player_buffer.remove(player).unwrap());
        }
        buffer
    } else { player_buffer };

    let mut results = Vec::new();
    let (tx, rx) = unbounded();

    // convert to tensor & process data
    for player in player_buffer {
        let mut look_tensors = Vec::new();
        let mut position_tensors = Vec::new();

        let steam_id = player.0;
        let mut player_info: VecDeque<Player> = VecDeque::new();

        for player in player.1 {
            player_info.push_back(player);
        }

        let mut finished = false;

        let mut batches = 0;
        let mut last_movement_batch = Vec::new();
        let mut last_look_batch = Vec::new();
        // println!("Working On: {steam_id}");
        while !finished {
            // let batch_start = Instant::now();
            // println!("Working on batch: {batches}");
            let mut c1: Vec<f64> = Vec::new(); 
            let mut c2: Vec<f64> = Vec::new(); 

            let mut c3: Vec<Vec<f64>> = Vec::new();

            // let total = player_info.len();

            for player in &player_info.clone() { // .into_iter().enumerate()
                if !( (c1.len() + c2.len())  >= ANGLE_INPUT_SIZE as usize) {
                    c1.push(player.view_angle as f64);
                    c2.push(player.pitch_angle as f64);

                    player_info.pop_front();
                }

                let position = player.position;

                let x = position.x as f64;
                let y = position.y as f64;
                let z = position.z as f64;
                let pos =  vec![x, y, z];
                
                c3.push(pos);

            }
            // let analysed_percent = (total as f64 / total as f64) * 100.0;
            // println!("{analysed_percent}% of {steam_id} user is being analysed");

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

            while !(look_buffer.len() >= ANGLE_INPUT_SIZE as usize) { look_buffer.push(0.0);finished = true; }

            while !(movement_buffer.len() >= MOVEMENT_INPUT_SIZE as usize) { movement_buffer.push(0.0);finished = true; }

            if last_movement_batch==movement_buffer && last_look_batch==look_buffer {
                // println!("Ending Batch Early! (Duplicate data!)");
                finished = true;
            }

            last_movement_batch = movement_buffer.clone();
            last_look_batch = look_buffer.clone();

            // println!("{movement_buffer:?}");

            let look_tensor = Tensor::from_slice(&look_buffer).reshape(&[1, ANGLE_INPUT_SIZE]);
            let position_tensor = Tensor::from_slice(&movement_buffer).reshape(&[1, MOVEMENT_INPUT_SIZE]);
            
            look_tensors.push(look_tensor);
            position_tensors.push(position_tensor);

            // let elapsed_time = batch_start.elapsed().as_secs();
            // println!("Batch {batches} took {elapsed_time}s");
            batches += 1;
        }

        tx.send(PlayerDemoTensor {
            steam_id: steam_id.to_string(),
            look_tensors,
            position_tensors,
            cheating: false,
        }).unwrap();
    };

    let _elapsed_time = work_start.elapsed().as_secs();
    // println!("Work Ended! Took {elapsed_time}s");

    for _ in 0..rx.len() {
        results.push(rx.recv().unwrap());
    }

    // println!("{}",results.first().unwrap().look_tensors.first().unwrap());
    
    results
}

#[derive(Debug)]
/// Represents a Player as tensors (batch tensors of length defined in main.rs)
pub struct PlayerDemoTensor {
    pub steam_id: String,
    pub cheating: bool,
    pub position_tensors: Vec<Tensor>,
    pub look_tensors: Vec<Tensor>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Represents a Player as tensors (batch tensors of length defined in main.rs) - Save able to the file system
pub struct PlayerDemoTensorSaveAble {
    pub steam_id: String,
    pub cheating: bool,
    pub position_tensors: Vec<Vec<f64>>,
    pub look_tensors: Vec<Vec<f64>>
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

#[derive(Debug, Default, Clone)]
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

impl PlayerDemoTensor {
    pub fn to_saveable(self) -> PlayerDemoTensorSaveAble {
        PlayerDemoTensorSaveAble {
            steam_id: self.steam_id.clone(),
            position_tensors: self.position_tensors.into_iter().map(tensor_to_vec).collect(),
            look_tensors: self.look_tensors.into_iter().map(tensor_to_vec).collect(),
            cheating: self.cheating,
        }
    }

    pub fn from_saveable(data: PlayerDemoTensorSaveAble) -> Self {
        PlayerDemoTensor {
            steam_id: data.steam_id,
            position_tensors: data.position_tensors.into_iter().map(|data| Tensor::from_slice(&data) ).collect(),
            look_tensors: data.look_tensors.into_iter().map(|data| Tensor::from_slice(&data) ).collect(),
            cheating: data.cheating,
        }
    }
}

/// Seer's version of a demo file
/// TODO: Anon support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeerDemo {
    pub cheater_steam_id: Vec<String>,
    pub data: Vec<PlayerDemoTensorSaveAble>
}

fn save_replay(data: Vec<PlayerDemoTensor>, path: &str, cheater_steam_ids: &[String]) -> Result<()> {

    let mut buffer = Vec::new();
    for item in data {

        let item = item.to_saveable();
        buffer.push(item);
    }

    // let data: Vec<PlayerDemoTensorSaveAble> = data.into_iter().map(|player| player.to_saveable()).collect();
    
    let data = SeerDemo {
        data: buffer,
        cheater_steam_id: cheater_steam_ids.to_vec(),
    };
    let data = serde_json::to_string(&data).unwrap();
    fs::write(path, data).unwrap();

    Ok(())
}

impl Clone for PlayerDemoTensor {
    fn clone(&self) -> Self {
        let mut position_tensors: Vec<Tensor> = Vec::new();
        for tensor in &self.position_tensors {
            let new_tensor = tensor.clone(&tensor);
            position_tensors.push(new_tensor)
        }

        let mut look_tensors: Vec<Tensor> = Vec::new();
        for tensor in &self.look_tensors {
            let new_tensor = tensor.clone(&tensor);
            look_tensors.push(new_tensor)
        }

        Self { steam_id: self.steam_id.clone(), position_tensors: position_tensors, look_tensors: look_tensors, cheating: self.cheating.clone() }
    }
}

pub fn hash_data(data: &[u8]) -> String { hex_digest(Algorithm::SHA256, data) }