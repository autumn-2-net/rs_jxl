use jxl_decoder::api::{JxlParallelRunner, JxlParallelRunnerFun};
use rayon::prelude::*;

pub(super) struct Runner(Option<rayon::ThreadPool>);
impl Runner {
    pub(super) fn pool(&self) -> Option<&rayon::ThreadPool> {
        self.0.as_ref()
    }

    pub(super) fn new(threads: usize) -> crate::Result<Self> {
        let pool = if threads > 1 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(super::failure)?,
            )
        } else {
            None
        };
        Ok(Self(pool))
    }
}
impl JxlParallelRunner for Runner {
    fn run(
        &mut self,
        num: usize,
        fun: &JxlParallelRunnerFun<'_>,
    ) -> jxl_decoder::error::Result<()> {
        if let Some(pool) = &self.0 {
            pool.install(|| (0..num).into_par_iter().try_for_each(fun))
        } else {
            (0..num).try_for_each(fun)
        }
    }
    fn num_threads(&self) -> usize {
        self.0
            .as_ref()
            .map_or(1, rayon::ThreadPool::current_num_threads)
    }
}
