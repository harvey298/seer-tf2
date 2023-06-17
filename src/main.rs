use std::{fs, path::Path};


use tch::{Tensor, Device, Kind, nn::Module};
use tf_demo_parser::{Demo, demo::{parser::{RawPacketStream, MessageHandler, gamestateanalyser::GameStateAnalyser}, header::Header, message::Message, data::DemoTick, gamevent::GameEvent, vector::Vector}, DemoParser, ParserState, MessageType};

use crate::net::{movement::MovementDetector, Seer, TrainingData};

pub const MOVEMENT_INPUT_SIZE: i64 = 2000;
pub const ANGLE_INPUT_SIZE: i64 = 2000;

mod net;

fn main() {

    let args = std::env::args().collect::<Vec<String>>();

    let mut seer = Seer::new().unwrap();
    
    if !Path::new("./training_data/").exists() {
        fs::create_dir_all("./training_data/labels/").unwrap();
        fs::create_dir_all("./training_data/demos/").unwrap();
        
    } else if args.contains(&"--train".to_string()) {
        let mut training_data: Vec<TrainingData> = Vec::new();

        for file in fs::read_dir("./training_data/labels/").unwrap() {
            let file = file.unwrap();
            let filename = file.file_name();
            let data = fs::read_to_string(Path::new("./training_data/labels/").join(filename)).unwrap();
            let data: TrainingData = toml::from_str(&data).unwrap();

            if let Some(exclude) = data.exclude { if exclude { continue; } }

            training_data.push(data);
        }

        seer.train(&training_data, true).unwrap();
    }

    for arg in args {
        if arg.contains(&"--demo=".to_string()) {
            let demo_file = arg.split("=").nth(1).unwrap();
            seer.scan_demo(demo_file).unwrap();
        }
    } 

    println!("Finished");
}

