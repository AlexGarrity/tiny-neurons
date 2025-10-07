use csv;

trait ModelData<const INPUTS: usize, const OUTPUTS: usize> {
    fn to_input(&self) -> [f32; INPUTS];
    fn to_actual(&self) -> [f32; OUTPUTS];
}

struct Model<const INPUTS: usize, const OUTPUTS: usize, T: ModelData<INPUTS, OUTPUTS>> {
    inputs: [Neuron<1>; INPUTS],
    outputs: [Neuron<INPUTS>; OUTPUTS],
    data: std::marker::PhantomData<T>
}

impl<const INPUTS: usize, const OUTPUTS: usize, T: ModelData<INPUTS, OUTPUTS>> Model<INPUTS, OUTPUTS, T> {
    fn new() -> Self {
        Self {
            inputs: core::array::from_fn(|_| {Neuron::new()}),
            outputs: core::array::from_fn(|_| {Neuron::new()}),
            data: std::marker::PhantomData
        }
    }

    fn predict(&self, data: &T) -> [f32; OUTPUTS] {
        let mut predictions = [0.0; INPUTS];
        let input = data.to_input();
        for i in 0..INPUTS {
            predictions[i] = self.inputs[i].predict([input[i]])
        }
        
        let mut outputs = [0.0; OUTPUTS];
        for i in 0..OUTPUTS {
            outputs[i] = self.outputs[i].predict(predictions);
        }
        
        outputs
    }

    fn error(&self, data: &T) -> f32 {
        let predictions = self.predict(data);
        let actual = data.to_actual();
        let mut error = 0.0;
        for i in 0..OUTPUTS {
            error += predictions[i] - actual[i];
        }

        error
    }

    fn loss(&self, data: &T) -> f32 {
        let predictions = self.predict(data);
        let actual = data.to_actual();
        let mut loss = 0.0;
        for i in 0..OUTPUTS {
            let x = predictions[i] - actual[i];
            loss += x * x * 0.5;
        }
        
        loss
    }

    fn step(&mut self, data: &T, lr: f32) {
        let mut predictions = [0.0; INPUTS];
        let input = data.to_input();
        for i in 0..INPUTS {
            predictions[i] = self.inputs[i].predict([input[i]]);
        }

        let error = self.error(data);

        for i in 0..OUTPUTS {
            self.outputs[i].update(predictions, lr, error);
        }
        for i in 0..INPUTS {
            self.inputs[i].update([input[i]], lr, error);
        }
    }
}

struct Neuron<const N: usize> {
    weight: [f32; N],
    bias: f32
}

impl<const N: usize> Neuron<N> {
    fn new() -> Self {
        let mut weight = [0.0; N];
        for i in 0..N {
            weight[i] = rand::random_range(0.0..1.0);
        }

        Self {
            weight,
            bias: 0.0
        }
    }

    fn predict(&self, v: [f32; N]) -> f32 {
        let mut x = 0.0;
        for i in 0..N {
            x += self.weight[i] * v[i];
        }

        x + self.bias
    }

    #[allow(unused)]
    fn error(&self, v: [f32; N], actual: f32) -> f32 {
        self.predict(v) - actual
    }

    #[allow(unused)]
    fn loss(&self, v: [f32; N], actual: f32) -> f32 {
        let x = self.predict(v) - actual;
        0.5 * x * x
    }

    fn update(&mut self, v: [f32; N], lr: f32, error: f32) {
        let db = error;
        self.bias = self.bias - lr * db;

        for i in 0..N {
            let dw = error * v[i];
            self.weight[i] -= lr * dw;
        }
    }

    #[allow(unused)]
    fn step(&mut self, v: [f32; N], actual: f32, lr: f32) {
        let error = self.error(v, actual);
        let db = error;

        self.bias = self.bias - lr * db;

        for i in 0..N {
            let dw = error * v[i];
            self.weight[i] -= lr * dw;
        }
    }
}

#[allow(dead_code)]
struct Game {
    rank: f32,
    name: String,
    platform: String,
    year: i32,
    genre: String,
    publisher: String,
    na_sales: f32,
    eu_sales: f32,
    jp_sales: f32,
    other_sales: f32,
    global_sales: f32
}

impl ModelData<4, 1> for Game {
    fn to_input(&self) -> [f32; 4] {
        [
            self.eu_sales,
            self.jp_sales,
            self.na_sales,
            self.other_sales
        ]
    }

    fn to_actual(&self) -> [f32; 1] {
        [ self.global_sales ]
    }
}

fn load_data() -> Vec<Game> {
    // vgsales.csv from Kaggle
    // https://www.kaggle.com/datasets/gregorut/videogamesales
    let file = std::fs::File::open("/home/alex/Development/neuron/vgsales.csv").unwrap();
    let mut reader = csv::Reader::from_reader(file);

    reader
        .records()
        .filter_map(|record| {
            if let Ok(r) = record {
                Some(
                    Game {
                        rank: r.get(0).unwrap().parse::<f32>().unwrap(),
                        name: String::from(r.get(1).unwrap()),
                        platform: String::from(r.get(2).unwrap()),
                        year: r.get(3).unwrap().parse().unwrap(),
                        genre: String::from(r.get(4).unwrap()),
                        publisher: String::from(r.get(5).unwrap()),
                        na_sales: r.get(6).unwrap().parse().unwrap(),
                        eu_sales: r.get(7).unwrap().parse().unwrap(),
                        jp_sales: r.get(8).unwrap().parse().unwrap(),
                        other_sales: r.get(9).unwrap().parse().unwrap(),
                        global_sales: r.get(10).unwrap().parse().unwrap(),
                    }
                )
            } else {
                None
            }
        })
        .collect::<Vec<Game>>()
}

fn average_loss(model: &Model<4,1, Game>, data: &Vec<Game>) -> f32 {
    let total_loss =
        data.iter().fold(0.0, |acc, game| {
            acc + model.loss(game)
        });
    total_loss / (data.len() as f32)
}

fn main() {
    let data = load_data();
    let learning_rate = 1e-5;

    let mut model = Model::<4, 1, Game>::new();

    for epoch in 1..=1000 {
        for game in &data {
            model.step(game, learning_rate);
        }

        let average_loss = average_loss(&model, &data);
        if !average_loss.is_normal() {
            panic!("Loss is not normal");
        }
        println!("Epoch {:5}, LR: {:0.2e}, Loss: {:2.8}", epoch, learning_rate, average_loss);
    }

    for i in 0..5 {
        println!("Prediction for {} sales: {} / {}", data[i].name, model.predict(&data[i])[0], data[i].global_sales as f32);
    }
    
}
