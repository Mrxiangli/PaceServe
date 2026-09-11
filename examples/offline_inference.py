import datetime
import ray
import os

from sarathi.benchmark.config import BenchmarkConfig
from sarathi.config import ModelConfig, ParallelConfig, PaceSchedulerConfig, MetricsConfig, ReplicaConfig, InstanceConfig, ControllerConfig, VllmSchedulerConfig
from sarathi import SamplingParams
from sarathi.disutil.mix_instance import MixInstance
from sarathi.disutil.prefill_instance import PrefillInstance
from sarathi.disutil.decode_instance import DecodeInstance
from sarathi.disutil.hp_instance import HPInstance
from sarathi.disutil.lp_instance import LPInstance
from sarathi.utils import get_ip
from sarathi.utils.signal_actor import SignalActor

from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

ray.init()

INSTANCE_CLASS = {
    'prefill': PrefillInstance,
    'decode': DecodeInstance,
    'mix': MixInstance,
    'high_priority': HPInstance,
    'low_priority': LPInstance
}

BASE_OUTPUT_DIR = "./offline_inference_output"

# Sample prompts.
prompts = [
    "The immediate reaction in some circles of the archeological community was that the accuracy of our dating was insufficient to make the extraordinary claim that humans were present in North America during the Last Glacial Maximum. But our targeted methodology in this current research really paid off, said Jeff Pigati, USGS research geologist and co-lead author of a newly published study that confirms the age of the White Sands footprints. The controversy centered on the accuracy of the original ages, which were obtained by radiocarbon dating. The age of the White Sands footprints was initially determined by dating seeds of the common aquatic plant Ruppia cirrhosa that were found in the fossilized impressions. But aquatic plants can acquire carbon from dissolved carbon atoms in the water rather than ambient air, which can potentially cause the measured ages to be too old. Even as the original work was being published, we were forging ahead to test our results with multiple lines of evidence, said Kathleen Springer, USGS research geologist and co-lead author on the current Science paper. We were confident in our original ages, as well as the strong geologic, hydrologic, and stratigraphic evidence, but we knew that independent chronologic control was critical.",
    "The breakthrough technique developed by University of Oxford researchers could one day provide tailored repairs for those who suffer brain injuries. The researchers demonstrated for the first time that neural cells can be 3D printed to mimic the architecture of the cerebral cortex. The results have been published today in the journal Nature Communications. Brain injuries, including those caused by trauma, stroke and surgery for brain tumours, typically result in significant damage to the cerebral cortex (the outer layer of the human brain), leading to difficulties in cognition, movement and communication. For example, each year, around 70 million people globally suffer from traumatic brain injury (TBI), with 5 million of these cases being severe or fatal. Currently, there are no effective treatments for severe brain injuries, leading to serious impacts on quality of life. Tissue regenerative therapies, especially those in which patients are given implants derived from their own stem cells, could be a promising route to treat brain injuries in the future. Up to now, however, there has been no method to ensure that implanted stem cells mimic the architecture of the brain.",
    "Hydrogen ions are the key component of acids, and as foodies everywhere know, the tongue senses acid as sour. That's why lemonade (rich in citric and ascorbic acids), vinegar (acetic acid) and other acidic foods impart a zing of tartness when they hit the tongue. Hydrogen ions from these acidic substances move into taste receptor cells through the OTOP1 channel. Because ammonium chloride can affect the concentration of acid -- that is, hydrogen ions -- within a cell, the team wondered if it could somehow trigger OTOP1. To answer this question, they introduced the Otop1 gene into lab-grown human cells so the cells produce the OTOP1 receptor protein. They then exposed the cells to acid or to ammonium chloride and measured the responses. We saw that ammonium chloride is a really strong activator of the OTOP1 channel, Liman said. It activates as well or better than acids. Ammonium chloride gives off small amounts of ammonia, which moves inside the cell and raises the pH, making it more alkaline, which means fewer hydrogen ions.",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=300)

output_dir = f"{BASE_OUTPUT_DIR}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

replica_config = ReplicaConfig(
    output_dir=output_dir,
)

model_config = ModelConfig(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    download_dir="/scratch/gilbreth/li2068/asplos/sarathi-serve/downloaded_model/"
)

scheduler_config = VllmSchedulerConfig(
    max_num_seqs = 128,
    regression_model_path = './regression_weights/2025-03-06-12:30'
)
ready_signal = SignalActor.remote()
stop_signal = SignalActor.remote()
instance_config = InstanceConfig()
tp, pp = instance_config.instance_parallel[0]
instance_parallel_config = ParallelConfig(pp, tp)
instance_config.resource_mapping = [(f"node:{get_ip()}",0)]

controller_config = ControllerConfig()

config = BenchmarkConfig()
config.model_config = model_config
config.scheduler_config = scheduler_config
config.instance_config = instance_config
config.controller_config = controller_config

instance_impl = INSTANCE_CLASS[instance_config._type]

instance_runner = instance_impl.options(
                num_cpus=12,
                max_concurrency = 4,
                ).remote(0, 0, config, instance_parallel_config, instance_config, ready_signal, stop_signal)

instance_runner._set_ref.remote(instance_runner)
instance_runner.run.remote()
ray.get(ready_signal.wait.remote())

for prompt in prompts:
    instance_runner._offline_inference.remote(prompt, sampling_params)
    
while not ray.get(instance_runner._check_output_finished.remote(len(prompts))):
    pass

outputs = ray.get(instance_runner._get_inference_result.remote())

for output in outputs:
    prompt = output.prompt
    generated_text = output.response
    print("===========================================================")
    print(f"Prompt: {prompt!r}")
    print("-----------------------------------------------------------")
    print(f"Generated text: {generated_text!r}")
    print("===========================================================")
