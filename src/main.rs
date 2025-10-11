use crate::layer::Layer;

trait Data<const INPUTS: usize, const OUTPUTS: usize> {
    fn load_data(path: &str) -> Vec<Self>;
    fn to_input(&self) -> [f32; INPUTS];
    fn to_actual(&self) -> [f32; OUTPUTS];
}

mod layer {
    pub trait Layer<const INPUT_SIZE: usize, const SIZE: usize> {
        fn new(activation: fn(&[f32; SIZE]) -> [f32; SIZE]) -> Self;
        fn predict(&self, input: &[f32; INPUT_SIZE]) -> [f32; SIZE];
        fn error(&self, input: &[f32; INPUT_SIZE], actual: &[f32; SIZE]) -> f32;
        fn loss(&self, input: &[f32; INPUT_SIZE], actual: &[f32; SIZE]) -> f32;
        fn update(&mut self, input: &[f32; INPUT_SIZE], lr: f32, error: f32);
    }

    pub struct Dense<const INPUTS: usize, const SIZE: usize> {
        neurons: [crate::Neuron<INPUTS>; SIZE],
        activation: fn(&[f32; SIZE]) -> [f32; SIZE]
    }

    impl<const INPUTS: usize, const SIZE: usize> Layer<INPUTS, SIZE> for Dense<INPUTS, SIZE> {
        fn new(activation: fn(&[f32; SIZE]) -> [f32; SIZE]) -> Self {
            Self {
                neurons: core::array::from_fn(|_| crate::Neuron::<INPUTS>::new()),
                activation
            }
        }

        fn predict(&self, input: &[f32; INPUTS]) -> [f32; SIZE] {
            let p = core::array::from_fn(|i| self.neurons[i].predict(input));
            (self.activation)(&p)
        }

        fn error(&self, input: &[f32; INPUTS], actual: &[f32; SIZE]) -> f32 {
            let mut total = 0.0;
            for i in 0..SIZE {
                total += self.neurons[0].predict(input) - actual[i]
            }
            total
        }

        fn loss(&self, input: &[f32; INPUTS], actual: &[f32; SIZE]) -> f32 {
            let predictions = self.predict(input);
            let mut total = 0.0;

            for i in 0..SIZE {
                let x = predictions[i] - actual[i];
                total += x * x * 0.5;
            }
            total
        }

        fn update(&mut self, input: &[f32; INPUTS], lr: f32, error: f32) {
            for i in 0..SIZE {
                self.neurons[i].update(input, lr, error);
            }
        }
    }

    pub type Input<const SIZE: usize> = Dense<SIZE, SIZE>;
}

mod activation {
    pub fn linear<const SIZE: usize>(v: &[f32; SIZE]) -> [f32; SIZE] {
        *v
    }

    pub fn relu<const SIZE: usize>(v: &[f32; SIZE]) -> [f32; SIZE] {
        core::array::from_fn(|i| { f32::max(0.0, v[i]) })
    }
}

struct Neuron<const N: usize> {
    weight: [f32; N],
    bias: f32,
}

impl<const N: usize> Neuron<N> {
    fn new() -> Self {
        let weight = core::array::from_fn(|_| rand::random_range(0.0..1.0));
        Self { weight, bias: 0.0 }
    }

    fn predict(&self, input: &[f32; N]) -> f32 {
        let mut x = 0.0;
        for (i, v) in input.iter().enumerate().take(N) {
            x += self.weight[i] * v;
        }

        x + self.bias
    }

    fn update(&mut self, input: &[f32; N], lr: f32, error: f32) {
        let db = error;
        self.bias -= lr * db;

        for (i, v) in input.iter().enumerate().take(N) {
            let dw = error * v;
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
    global_sales: f32,
}

impl Data<4, 1> for Game {
    fn load_data(path: &str) -> Vec<Game> {
        // vgsales.csv from Kaggle
        // https://www.kaggle.com/datasets/gregorut/videogamesales
        let file = std::fs::File::open(path).unwrap();
        let mut reader = csv::Reader::from_reader(file);

        reader
            .records()
            .filter_map(|record| {
                if let Ok(r) = record {
                    Some(Game {
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
                    })
                } else {
                    None
                }
            })
        .collect::<Vec<Game>>()
    }

    fn to_input(&self) -> [f32; 4] {
        [
            self.eu_sales,
            self.jp_sales,
            self.na_sales,
            self.other_sales,
        ]
    }

    fn to_actual(&self) -> [f32; 1] {
        [self.year as f32]
    }
}

struct Model<const INPUT: usize, const OUTPUT: usize> {
    input: layer::Input<INPUT>,
    dense_1: layer::Dense<INPUT, 20>,
    dense_2: layer::Dense<20, 20>,
    output: layer::Dense<20, OUTPUT>
}

impl<const INPUT: usize, const OUTPUT: usize> Model<INPUT, OUTPUT> {
    fn new() -> Self {
        Self {
            input: layer::Input::new(activation::linear),
            dense_1: layer::Dense::new(activation::relu),
            dense_2: layer::Dense::new(activation::relu),
            output: layer::Dense::new(activation::linear),
        }
    }

    fn average_loss<T: Data<INPUT, OUTPUT>>(&self, data: &Vec<T>) -> f32 {
        let total =
            data.iter().fold(0.0, |acc, d| {
                acc + self.loss(d)
            });

        total / (data.len() as f32)
    }

    fn error<T: Data<INPUT, OUTPUT>>(&self, data: &T) -> f32 {
        let input = data.to_input();
        let actual = data.to_actual();

        let pi = self.input.predict(&input);
        let pd1 = self.dense_1.predict(&pi);
        let pd2 = self.dense_2.predict(&pd1);

        let error = self.output.error(&pd2, &actual);
        error
    }

    fn loss<T: Data<INPUT, OUTPUT>>(&self, data: &T) -> f32 {
        let input = data.to_input();
        let actual = data.to_actual();

        let pi = self.input.predict(&input);
        let pd1 = self.dense_1.predict(&pi);
        let pd2 = self.dense_2.predict(&pd1);

        let loss = self.output.loss(&pd2, &actual);
        loss
    }

    fn predict<T: Data<INPUT, OUTPUT>>(&self, data: &T) -> [f32; OUTPUT] {
        let input = data.to_input();

        let pi = self.input.predict(&input);
        let pd1 = self.dense_1.predict(&pi);
        let pd2 = self.dense_2.predict(&pd1);

        let prediction = self.output.predict(&pd2);
        prediction
    }

    fn step<T: Data<INPUT, OUTPUT>>(&mut self, data: &T, lr: f32) {
        let input = data.to_input();
        let actual = data.to_actual();

        let pi = self.input.predict(&input);
        let pd1 = self.dense_1.predict(&pi);
        let pd2 = self.dense_2.predict(&pd1);

        let error = self.output.error(&pd2, &actual);

        self.input.update(&input, lr, error);
        self.dense_1.update(&pi, lr, error);
        self.dense_2.update(&pd1, lr, error);
        self.output.update(&pd2, lr, error);
    }
}

fn main() {
    let data = Game::load_data("/home/alex/Development/neuron/vgsales.csv");
    let learning_rate = 1e-8;

    let mut model = Model::new();

    for epoch in 1..=10000 {
        for game in &data {
            model.step(game, learning_rate);
        }

        let average_loss = model.average_loss(&data);
        if !average_loss.is_normal() {
            panic!("Loss is not normal ({average_loss})");
        }
        println!(
            "Epoch {:5}, LR: {:0.2e}, Loss: {:2.8}",
            epoch, learning_rate, average_loss
        );
    }

    for d in data.iter().take(5) {
        println!(
            "Prediction for {} sales: {} / {}",
            d.name,
            model.predict(d)[0],
            d.to_actual()[0]
        );
    }
}
