use std::path::PathBuf;

use chumsky::prelude::*;
// fn parse() -> impl Parser<char Vec<>>
// fn models() -> impl Parser<char, Vec<super::ModelIO>> {
//     let model = super::ModelIO::parser();
//     let comma = char(',').skip_many1();
//     let models = model.sep_by(comma);
//     models
// }
pub enum ModelIOArgs {
    Path(PathBuf),
    Assert(PathBuf),
    InputType(super::DataTypes),
    OutputType(super::DataTypes),
}

// pub fn arg<'a, T: Clone + 'a, E: chumsky::Error<&'a T>>(
//     s: T,
// ) -> chumsky::primitive::Just<&'a [T], &'a [T], E> {
//     just(&[s])
// }
macro_rules! arg {
    ($s:expr) => {
        just::<&str, _, Simple<&str>>($s)
    };
}

fn models<'a>() -> impl Parser<&'a str, Vec<super::ModelIO>, Error = Simple<&'a str>> {
    let assert = choice((arg!("--assert"), arg!("-a")))
        .then(path())
        .map(|(_, p)| p);
    let data_type = choice((
        arg!("f32").to(super::DataTypes::F32),
        arg!("u8").to(super::DataTypes::U8),
    ));
    let input_type = choice((arg!("--input-type"), arg!("-i")))
        .then(data_type)
        .map(|(_, t)| t);
    let output_type = choice((arg!("--output-type"), arg!("-o")))
        .then(choice((
            arg!("f32").to(super::DataTypes::F32),
            arg!("u8").to(super::DataTypes::U8),
        )))
        .map(|(_, t)| t);
    let args = choice((
        // path.map(|p| ModelIOArgs::Path(p)),
        assert.map(|p| ModelIOArgs::Assert(p)),
        input_type.map(|t| ModelIOArgs::InputType(t)),
        output_type.map(|t| ModelIOArgs::OutputType(t)),
    ))
    .repeated();
    let mios = path().then(args).map(|(p, margs)| {
        let mut mio = super::ModelIO::default();
        mio.path = p;
        margs.into_iter().for_each(|arg| match arg {
            ModelIOArgs::Path(p) => mio.path = p,
            ModelIOArgs::Assert(p) => mio.assert = Some(p),
            ModelIOArgs::InputType(t) => mio.input_type = t,
            ModelIOArgs::OutputType(t) => mio.output_type = t,
        });
        mio
    });
    mios.repeated()
}

#[derive(Debug, Clone)]
pub enum Flags {
    Verbose,
    Warmup(u8),
    Output(PathBuf),
    Exec,
}
fn flags<'a>() -> impl Parser<&'a str, Vec<Flags>, Error = Simple<&'a str>> {
    choice((
        choice((arg!("--verbose"), arg!("-v"))).to(Flags::Verbose),
        choice((arg!("--warmup"), arg!("-w")))
            .ignore_then(any().from_str().unwrapped())
            .map(Flags::Warmup),
    ))
    .repeated()
}

fn path<'i>() -> impl Parser<&'i str, PathBuf, Error = Simple<&'i str>> {
    any().map(|c| PathBuf::from(c))
}

impl super::Cli {
    pub fn try_from_env() -> super::Result<Self> {
        // let args: Vec<_> = std::env::args()
        //     // .enumerate()
        //     // .map(|(i, a)| (a, i..i + 1))
        //     .collect();
        // let args_str: Vec<_> = args
        //     .iter()
        //     // .enumerate()
        //     // .map(|(i, item)| (item.as_str(), i..i + 1))
        //     .map(|i| i.as_str())
        //     .collect();
        let args = std::env::args().collect::<Vec<_>>();
        let args_str = args.iter().map(|i| i.as_str()).collect::<Vec<_>>();

        let mio = path()
            .then(choice((models().to(()), flags().to(()))))
            .parse(args_str);

        // let mio = super::ModelIO::parse().parse(args_str.as_slice());
        dbg!(mio.unwrap());
        todo!()
    }
}
#[derive(Debug, Clone, ValueEnum, Default)]
pub enum DataTypes {
    #[default]
    F32,
    U8,
}

#[derive(Debug, Clone, Args, Default)]
pub struct ModelIO {
    path: PathBuf,
    #[clap(short, long)]
    assert: Option<PathBuf>,
    #[clap(short, long, default_value = "f32")]
    input_type: DataTypes,
    #[clap(short, long, default_value = "f32")]
    output_type: DataTypes,
}
impl AsRef<Path> for ModelIO {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}
