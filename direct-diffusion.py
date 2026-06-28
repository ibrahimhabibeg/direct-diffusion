# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "einops==0.8.2",
#     "matplotlib==3.11.0",
#     "numpy==2.5.0",
#     "pillow==12.2.0",
#     "pybtex==0.26.1",
#     "torch==2.12.1",
#     "torchvision==0.27.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="Don't predict the noise, just go for the data",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Diffusion Models Should Just Go for the Data
    """).center()
    return


@app.cell(hide_code=True)
def _():
    # Setup
    import os
    import math
    import marimo as mo
    import torch
    from torch import nn
    import argparse
    from PIL import Image
    import numpy as np
    import torchvision
    import matplotlib.pyplot as plt
    import warnings
    from pybtex.database import parse_file
    import requests
    # import shutil
    import os
    import sys
    import subprocess
    import marimo as mo

    repo_url = "https://github.com/lth14/jit.git"
    clone_dir = "jit_repo"

    if not os.path.exists(clone_dir):
        subprocess.run(["git", "clone", repo_url, clone_dir], check=True)

    repo_path = os.path.abspath(clone_dir)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from denoiser import Denoiser

    shared_url = "https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/AGlp4FoN0cIF8nMbS4DN7Ns/jit-b-16/checkpoint-last.pth?rlkey=14gjrblmljewpl6ygxzlr3njm&dl=1"
    file_name = "checkpoint-last.pth"
    target_directory = "checkpoints"
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    full_local_path = os.path.join(target_directory, file_name)

    if not os.path.exists(full_local_path):
        response = requests.get(shared_url)
        if response.status_code == 200:
            with open(full_local_path, "wb") as file:
                file.write(response.content)
        else:
            print("Failed to download pretrained model")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0);
    return (
        Denoiser,
        Image,
        argparse,
        device,
        math,
        mo,
        nn,
        np,
        os,
        parse_file,
        plt,
        requests,
        torch,
        torchvision,
    )


@app.cell(hide_code=True)
def _(parse_file, requests):
    def create_citation_dict(keys, bib_file):

        bib_data = parse_file(bib_file)
        final_citations = {}

        for key in keys:
            if key not in bib_data.entries:
                print(f"WARNING: Key '{key}' not found in {bib_file}")
                continue

            entry = bib_data.entries[key]        
            year = entry.fields.get('year', 'n.d.')
            authors = entry.persons.get('author', [])
            last_names = [
                " ".join(person.last_names).replace('{', '').replace('}', '') 
                for person in authors if person.last_names
            ]

            if len(last_names) == 0:
                author_str = "Unknown Author"
            elif len(last_names) == 1:
                author_str = last_names[0]
            elif len(last_names) == 2:
                author_str = f"{last_names[0]} & {last_names[1]}"
            else:
                author_str = f"{last_names[0]} et al."

            final_citations[key] = {
                'ordinary': f"[({author_str}, {year})](#references)",
                'narrative': f"[{author_str} ({year})](#references)"
            }

        return final_citations

    bib_file_remote = "https://raw.githubusercontent.com/ibrahimhabibeg/direct-diffusion/refs/heads/main/references.bib"
    bib_file = 'references.bib'
    bib_keys = ['li_back_2026', 'bansal_cold_2022', 'ho_denoising_2020']

    bib_response = requests.get(bib_file_remote)
    if bib_response.status_code == 200:
        with open(bib_file, 'wb') as f:
            f.write(bib_response.content)
    else:
        print(f"Failed to download the bib file. Status code: {bib_response.status_code}")

    citations_dict = create_citation_dict(bib_keys, bib_file)

    def format_bibliography_authors(authors):
        """Helper function to format authors as 'Last, F., & Last, F.'"""
        if not authors:
            return "Unknown Author"

        formatted_authors = []
        for person in authors:
            # Extract last name and strip BibTeX braces
            last = " ".join(person.last_names).replace('{', '').replace('}', '')

            # Extract first initials
            first_initials = [n[0] + "." for n in person.first_names if n]
            first = " ".join(first_initials).replace('{', '').replace('}', '')

            if first:
                formatted_authors.append(f"{last}, {first}")
            else:
                formatted_authors.append(last)

        # Join them logically based on the number of authors
        if len(formatted_authors) == 1:
            return formatted_authors[0]
        elif len(formatted_authors) == 2:
            return f"{formatted_authors[0]} & {formatted_authors[1]}"
        else:
            # For 3+ authors: Last, F., Last, F., & Last, F.
            return ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"


    def generate_markdown_bibliography(keys, bib_file):
        """
        Parses a .bib file and generates a Markdown-formatted reference list.
        """
        bib_data = parse_file(bib_file)
        md_lines = []

        for key in keys:
            if key not in bib_data.entries:
                print(f"WARNING: Key '{key}' not found in {bib_file}")
                continue

            entry = bib_data.entries[key]

            # 1. Format Authors
            authors = entry.persons.get('author', [])
            author_str = format_bibliography_authors(authors)

            # 2. Extract Year and Title
            year = entry.fields.get('year', 'n.d.')
            title = entry.fields.get('title', 'Untitled').replace('{', '').replace('}', '')

            # 3. Extract Publisher / Journal (Fallback logic)
            source = entry.fields.get('journal', 
                     entry.fields.get('publisher', 
                     entry.fields.get('booktitle', '')))

            # Markdown italics for the source publication
            source_str = f" *{source}*." if source else ""

            # 4. Extract Links (Prioritize DOI, fallback to URL)
            doi = entry.fields.get('doi', '')
            url = entry.fields.get('url', '')
            link_str = ""

            if doi:
                # Convert raw DOI to a clickable link
                link_str = f" https://doi.org/{doi}"
            elif url:
                link_str = f" {url}"

            # 5. Assemble the final Markdown bullet point
            md_line = f"* {author_str} ({year}). {title}.{source_str}{link_str}"
            md_lines.append(md_line.strip())

        return "\n".join(md_lines)

    return bib_file, bib_keys, citations_dict, generate_markdown_bibliography


@app.cell(hide_code=True)
def _(citations_dict, mo):
    mo.md(f"""
    Released earlier this year, "Back to Basics: Let Denoising Generative Models Denoise" {citations_dict['li_back_2026']['ordinary']} paper challenged what has been the norm for years for building diffusion models: using neural networks for noise prediction.

    This blog will focus on that paper. The authors have made multiple contributions in the paper, and this blog will focus on only one aspect of it: diffusion models directly predicting the datapoints. The goal of this blog is to clearly explain how diffusion models work and explain the idea suggested by the authors of the paper. We will go over the maths of diffusion, build a minimal model, train it on a toy dataset, and visualize the output.

    By the end of reading this, you will hopefully gain a better understanding of diffusion models and how $\epsilon$-prediction differs from $x$-prediction, and you will gain the ability to build them on your own from scratch.

    First things first, before focusing on the paper, let's review diffusion models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diffusion Models Recap

    In all generative models, we typically want to learn a distribution $p_{data}$ over the data $x$ from the training set $\{x_1, x_2, ..., x_N\}$. We also want a technique to be able to sample new data points from the learned distribution.

    In diffusion models, we start from a simple distribution (like Gaussian noise), and we learn a technique to transform a sample from that distribution $\epsilon \sim \mathcal{N}(0, I)$ into a sample from the data distribution $x \sim p_{data}$.
    """)
    return


@app.cell(hide_code=True)
def _(citations_dict, mo):
    mo.md(f"""
    ## The Forward Process

    To learn how to transform, we first define the **forward process**. You can think of this as a *noising* process. The goal of this process is to *destroy information*. That is, we take in a sample from the data distribution $x \sim p_{{data}}$ (this is a value from the training set) and transform it to a sample from the simple one (e.g., $N(0, I)$). The most common approach for denoising was popularized by DDPM {citations_dict["ho_denoising_2020"]["ordinary"]}: adding Gaussian noise.

    The noising process occurs over multiple steps. We define a set of intermediate levels between $x$ and $\epsilon$ of the data as $z_t$ where $t \in [0, 1]$ is the time step. So to reclarify, $z_t$ is some sort of a mix between the original data $x$ and the noise $\epsilon$. The forward process is defined as: $z_t = a_tx + b_t\epsilon$ where $a_t$ and $b_t$ are the noise schedules. For simplicity, we will just use a linear schedule, which gives us:

    $$
    z_t = tx + (1-t)\epsilon
    $$

    So at time step $t=1$, we have the original data $z_1 = x$, and at time step $t=0$, we have the noise $z_0 = \epsilon$. [[1]](#footnotes)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Reverse Process

    The **reverse process** does the opposite of the forward process. You can think of this as a *denoising* process. It takes in a sample from the simple distribution $\epsilon \sim \mathcal{N}(0, I)$ and transforms it into a sample from the data distribution $x \sim p_{data}$.

    In typical diffusion models like DDPM, we learn a function $\epsilon_{\Theta}$ that can predict the noise $\epsilon$ given a sample $z_t$ and the time step $t$. This function is typically a neural network. Once we have learned this function, we can use it to denoise a sample from the simple distribution $\epsilon \sim \mathcal{N}(0, I)$ to obtain a sample from the data distribution $x \sim p_{data}$ by looping over the time steps and removing the predicted noise at each step.

    Now that we have understood the two core processes of diffusion models, let's see how everything fits together for training and sampling.

    ## The Training Algorithm

    The goal of the training algorithm is to learn the function $\epsilon_{\Theta}$ that can predict the noise $\epsilon$ given a sample $z_t$ and the time step $t$. This is done by applying the forward process on a batch of data points from the training set to obtain $z_t$, and then using the neural network to predict the noise $\epsilon$. The loss function is typically the mean squared error between the predicted noise and the actual noise added.

    Let's summarize the training algorithm in steps (net is the neural network that predicts the noise):

    ```psuedocode
    for each training iteration:
        x ~ p_{data}  # Sample a batch of data points from the training set
        eps ~ N(0, I)  # Sample noise from the simple distribution. Must be the same shape as x
        t ~ Uniform(0, 1)  # Sample a time step from the uniform distribution
        z = t * x + (1-t) * eps  # Add noise to the data points to obtain z_t
        eps_pred = net(z, t)  # Use the neural network to predict
        loss = MSE(eps_pred, eps)  # Compute the loss between the predicted noise and the actual noise
        Update the neural network parameters using backpropagation and an optimizer
    ```

    ## The Sampling Algorithm

    The goal of sampling is to create new data points from nothing but noise. Here, the reverse process is applied. We use $\epsilon_{\Theta}$ to predict the noise $\epsilon$ given a sample $z_t$ and the time step $t$. We then remove the predicted noise from the sample to obtain a new updated one.

    This process is performed iteratively over $T$ steps. We start with $z_0 = \epsilon$ and apply the reverse process to obtain $z_1$, then use $z_1$ to obtain $z_2$, and so on, until we reach $z_T = x$.

    ```psuedocode
    z_0 ~ N(0, I)  # Sample noise from the simple distribution
    t = linspace(0, 1, T)  # Create a list of time steps from 0 to 1
    for i in range(T):
        eps_pred = net(z_i, t[i])  # Use the neural network to predict
        x_pred = (z_i - (1-t[i]) * eps_pred) / t[i]  # Compute the predicted data point based on eps_pred
        z_{i+1} = t[i+1] * x_pred + (1-t[i+1]) * eps_pred  # Update the sample for the next step]
    return z_T  # Return the final sample from the data distribution
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizing Diffusion Models

    We now understand that diffusion is a process that learns how to reconstruct images from noise. If this is your first time learning about them, I know the idea could be a bit abstract. So, let's visualize the process by creating a figure that shows us how the noise progresses until it becomes a datapoint.

    The figure below shows the trajectory of the denoising process. The leftmost image is the noise we sample from the simple distribution ($\epsilon \sim \mathcal{N}(0, I)$ this case), and the rightmost image is the final generated image. The images in between are the intermediate steps of the denoising process. The final image is generated through 50 steps of denoising. In each step, a neural network is used to restore the image from noisy input.

    But let me clarify something here. The network used for this visualization is the pretrained model provided by the authors of the paper. I know I've told you that their method is a bit different from the standard diffusion models, but for the sake of this visualization, the differences don't matter. In the next section, we will focus on the differences and the core message of the paper.

    Since we will later be building a diffusion model anyway, I don't want you to focus on the code now, as we will later get to it.
    """)
    return


@app.cell(hide_code=True)
def _(argparse):
    def create_args():
        parser = argparse.ArgumentParser('JiT', add_help=False)

        # architecture
        parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                            help='Name of the model to train')
        parser.add_argument('--img_size', default=256, type=int, help='Image size')
        parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
        parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

        # training
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                            help='Epochs to warm up LR')
        parser.add_argument('--batch_size', default=128, type=int,
                            help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='Learning rate (absolute)')
        parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                            help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
        parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                            help='Minimum LR for cyclic schedulers that hit 0')
        parser.add_argument('--lr_schedule', type=str, default='constant',
                            help='Learning rate schedule')
        parser.add_argument('--weight_decay', type=float, default=0.0,
                            help='Weight decay (default: 0.0)')
        parser.add_argument('--ema_decay1', type=float, default=0.9999,
                            help='The first ema to track. Use the first ema for sampling by default.')
        parser.add_argument('--ema_decay2', type=float, default=0.9996,
                            help='The second ema to track')
        parser.add_argument('--P_mean', default=-0.8, type=float)
        parser.add_argument('--P_std', default=0.8, type=float)
        parser.add_argument('--noise_scale', default=1.0, type=float)
        parser.add_argument('--t_eps', default=5e-2, type=float)
        parser.add_argument('--label_drop_prob', default=0.1, type=float)

        parser.add_argument('--seed', default=0, type=int)
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='Starting epoch')
        parser.add_argument('--num_workers', default=12, type=int)
        parser.add_argument('--pin_mem', action='store_true',
                            help='Pin CPU memory in DataLoader for faster GPU transfers')
        parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
        parser.set_defaults(pin_mem=True)

        # sampling
        parser.add_argument('--sampling_method', default='heun', type=str,
                            help='ODE samping method')
        parser.add_argument('--num_sampling_steps', default=50, type=int,
                            help='Sampling steps')
        parser.add_argument('--cfg', default=1.0, type=float,
                            help='Classifier-free guidance factor')
        parser.add_argument('--interval_min', default=0.0, type=float,
                            help='CFG interval min')
        parser.add_argument('--interval_max', default=1.0, type=float,
                            help='CFG interval max')
        parser.add_argument('--num_images', default=50000, type=int,
                            help='Number of images to generate')
        parser.add_argument('--eval_freq', type=int, default=40,
                            help='Frequency (in epochs) for evaluation')
        parser.add_argument('--online_eval', action='store_true')
        parser.add_argument('--evaluate_gen', action='store_true')
        parser.add_argument('--gen_bsz', type=int, default=256,
                            help='Generation batch size')

        # dataset
        parser.add_argument('--data_path', default='./data/imagenet', type=str,
                            help='Path to the dataset')
        parser.add_argument('--class_num', default=1000, type=int)

        # checkpointing
        parser.add_argument('--output_dir', default='./output_dir',
                            help='Directory to save outputs (empty for no saving)')
        parser.add_argument('--resume', default='',
                            help='Folder that contains checkpoint to resume from')
        parser.add_argument('--save_last_freq', type=int, default=5,
                            help='Frequency (in epochs) to save checkpoints')
        parser.add_argument('--log_freq', default=100, type=int)
        parser.add_argument('--device', default='cuda',
                            help='Device to use for training/testing')

        # distributed training
        parser.add_argument('--world_size', default=1, type=int,
                            help='Number of distributed processes')
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='URL used to set up distributed training')
        return parser

    args = create_args().parse_args([]) 
    return (args,)


@app.cell(hide_code=True)
def _(Denoiser, args, device, os, torch):
    def load_pretrained_model():
        model = Denoiser(args)
        checkpoint_path = "checkpoints/checkpoint-last.pth"
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        model = model.net
        model.to(device)
        return model

    pretrained_model = load_pretrained_model()
    return (pretrained_model,)


@app.cell(hide_code=True)
def _(Image, args, device, np, pretrained_model, torch):
    @torch.no_grad()
    def forward_sample(z, t, labels):
        # conditional
        x_cond = pretrained_model(z, t.flatten(), labels)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(args.t_eps)
        # unconditional
        x_uncond = pretrained_model(z, t.flatten(), torch.full_like(labels, args.class_num))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(args.t_eps)
        # cfg interval
        low, high = args.interval_min, args.interval_max
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, args.cfg, 1.0)
        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    def stepper(z, t, t_next, labels):
        v_pred = forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    def generate(z, label):
        labels = torch.tensor([label])
        labels = labels.to(device)

        timesteps = torch.linspace(0.0, 1.0, args.num_sampling_steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, 1, -1, -1, -1)
        frames = []
        for i in range(args.num_sampling_steps - 1):
            if i % 5 == 0:
                z_cpu = z.detach().cpu()[0]
                z_cpu = ((z_cpu + 1) / 2.0).clamp(0, 1)
                frames.append(z_cpu)
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        z = stepper(z, timesteps[-2], timesteps[-1], labels)
        z_cpu = z.detach().cpu()[0]
        z_cpu = ((z_cpu + 1) / 2.0).clamp(0, 1)
        frames.append(z_cpu)
        return frames, timesteps.reshape(-1).tolist()[::5]

    def get_trajectory():
        label = 10
        z = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        frames, timesteps = generate(z, label)
        frames = [Image.fromarray((frame * 255).clip(0, 255).numpy().astype(np.uint8).transpose(1, 2, 0)) for frame in frames]
        return frames, timesteps

    diffusion_trajectory_frames, diffusion_trajectory_timesteps = get_trajectory()
    return diffusion_trajectory_frames, diffusion_trajectory_timesteps


@app.cell(hide_code=True)
def _(mo):
    timestep_slider = mo.ui.slider(
        start=0,
        stop=10,
        step=1,
        value=10,
        label="**ODE Solver Step**",
        show_value=True
    )
    return (timestep_slider,)


@app.cell(hide_code=True)
def _(
    diffusion_trajectory_frames,
    diffusion_trajectory_timesteps,
    mo,
    timestep_slider,
):
    gallery_indices = [0, 3, 5, 7, 10]

    gallery_layout = mo.hstack(
        [
            mo.vstack([
                mo.md(f"**Step {i}**<br>*(t = {diffusion_trajectory_timesteps[i]:.2f})*"), 
                mo.image(diffusion_trajectory_frames[i], width=150)
            ], align="center") 
            for i in gallery_indices
        ],
        justify="space-between"
    )

    current_index = timestep_slider.value
    current_t = diffusion_trajectory_timesteps[current_index]

    # Render the entire combined UI
    mo.vstack([
        mo.md("### The Denoising Process"),
        mo.md("Below are 5 key snapshots from the 50-step trajectory recovering the image from pure noise to the clean data."),
        gallery_layout,

        mo.md("---"),

        mo.md("### Interactive Denoising"),
        mo.md(f"""
        Use the slider to explore 10 steps of the generation process. Currently displaying **Step {current_index}** *(t = {current_t:.2f})*.
        Notice how, as $t$ increases, information is gradually recovered, and the image becomes clearer.
        """),
        timestep_slider,
        mo.image(diffusion_trajectory_frames[current_index], width=512)
    ], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Core Idea of the Paper

    The paper we're focusing on today, "Back to Basics: Let Denoising Generative Models Denoise", has multiple contributions, but the core idea is simple: instead of predicting the noise added to the data at each timestep, we can predict the original data directly.

    The hypothesis they are making can be briefly summarized as: predicting data is easier than predicting noise. To keep this blog simple and focused, I will not go into the details of the hypothesis or the experiment they did to support it. Interested readers can read section 3.3 of the paper for an interesting experiment.

    Now, let's consider how the training and sampling algorithms would change if we were to predict the original data directly instead of predicting the noise. For the training, the loss function will need to be changed. This time we will compute the mean squared error between the predicted data and the actual data instead of the predicted noise and the actual noise. The rest of the training algorithm remains the same.

    ```psuedocode
    for each training iteration:
        x ~ p_{data}  # Sample a batch of data points from the training set
        eps ~ N(0, I)  # Sample noise from the simple distribution. Must be the same shape as x
        t ~ Uniform(0, 1)  # Sample a time step from the uniform distribution
        z = t * x + (1-t) * eps  # Add noise to the data points to obtain z_t
        x_pred = net(z, t)  # Use the neural network to predict
        loss = MSE(x_pred, x)  # Compute the loss between the predicted data and the actual data
        Update the neural network parameters using backpropagation and an optimizer
    ```


    For the sampling process, we will need to change the way $z_{i+1}$ is computed. Instead of using the predicted noise to compute $z_{i+1}$, we will use the predicted data. The rest of the sampling algorithm remains the same. [[2]](#footnotes)

    ```psuedocode
    z_0 ~ N(0, I)  # Sample noise from the simple distribution
    eps = z_0  # Store the initial noise for later use
    t = linspace(0, 1, T)  # Create a list of time steps from 0 to 1
    for i in range(T):
        x_pred =  net(z_i, t[i])
        z_{i+1} = t[i+1] * x_pred + (1-t[i+1]) * eps  # Update the sample for the next step]
    return z_T  # Return the final sample from the data distribution
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Implementing the New Methodology

    Now that we have an understanding of the new methodology, let's implement it. We will train the model on the MNIST dataset for simplicity, and to keep this blog focused, we will support only unconditional generation.

    Before we get into writing code, we need to figure out what components we need. We will need:

    - A **neural network** to predict the original data
    - A **data loader** to load batches of data from the MNIST dataset
    - A **training function** to train the neural network using the data loader
    - A **sampling function** to generate new data points from the trained neural network

    Let's take it step by step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The Neural Network

    The first component is the neural network. Notice that the diffusion framework treats the network as a black box. Thus, this isn't our focus, and the provided network probably isn't the best one.

    The network we will use is a simple UNet containing only 2 downsampling and upsampling blocks. This network will be conditioned on the time step but not on the class label.

    We will make the number of input channels an input parameter so that the code can easily be reused with other datasets containing RGB images, but the comments assume the dimensions of MNIST.
    """)
    return


@app.cell(hide_code=True)
def _(math, nn, torch):
    class SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, dim, scale=1000.0):
            super().__init__()
            self.dim = dim
            self.scale = scale

        def forward(self, t):
            """t: Continuous time tensor of shape [B, 1]"""
            device = t.device
            half_dim = self.dim // 2

            embeddings = math.log(10000.0) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

            t_scaled = t * self.scale
            embeddings = t_scaled * embeddings.unsqueeze(0)

            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
            return embeddings

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, time_emb_dim):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.act = nn.SiLU()
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        def forward(self, x, t_emb):
            h = self.act(self.conv1(x))
            time_val = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + time_val
            h = self.act(self.conv2(h))
            return h

    class UNet(nn.Module):
        def __init__(self, in_channels=1, time_emb_dim=128):
            super().__init__()
            self.time_emb_dim = time_emb_dim

            self.sinu_embed = SinusoidalTimeEmbedding(time_emb_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )

            self.down1 = DoubleConv(in_channels, 16, time_emb_dim)
            self.pool1 = nn.MaxPool2d(2)

            self.down2 = DoubleConv(16, 32, time_emb_dim)
            self.pool2 = nn.MaxPool2d(2)

            self.bottleneck = DoubleConv(32, 64, time_emb_dim)

            self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.up_conv1 = DoubleConv(64, 32, time_emb_dim) # 64 because of skip connection (32+32)

            self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
            self.up_conv2 = DoubleConv(32, 16, time_emb_dim) # 32 because of skip connection (16+16)

            self.out_conv = nn.Conv2d(16, in_channels, kernel_size=1)

        def forward(self, x, t):
            """
            x: Noisy input images of shape [B, 1, 28, 28]
            t: Continuous timesteps of shape [B, 1] (values between 0.0 and 1.0)
            """
            t_sinu = self.sinu_embed(t)            # [B, 128]
            t_emb = self.time_mlp(t_sinu)          # [B, 128]

            d1 = self.down1(x, t_emb)          # [B, 16, 28, 28]
            p1 = self.pool1(d1)                # [B, 16, 14, 14]

            d2 = self.down2(p1, t_emb)         # [B, 32, 14, 14]
            p2 = self.pool2(d2)                # [B, 32, 7, 7]

            bot = self.bottleneck(p2, t_emb)   # [B, 64, 7, 7]

            u1 = self.up1(bot)                 # [B, 32, 14, 14]
            u1 = torch.cat([u1, d2], dim=1)    # skip connection -> [B, 64, 14, 14]
            u1 = self.up_conv1(u1, t_emb)      # [B, 32, 14, 14]

            u2 = self.up2(u1)                  # [B, 16, 28, 28]
            u2 = torch.cat([u2, d1], dim=1)    # skip connection -> [B, 32, 28, 28]
            u2 = self.up_conv2(u2, t_emb)      # [B, 16, 28, 28]

            pred_clean_image = self.out_conv(u2) # [B, 1, 28, 28]

            return pred_clean_image

    return (UNet,)


@app.cell(hide_code=True)
def _(UNet, mo):
    number_of_parameters = sum(p.numel() for p in UNet().parameters())

    mo.md(f"""
    The model has **{number_of_parameters:,} parameters**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The Data Loader

    We will just use torchvision's MNIST dataset for downloading and reading the data. We will pass the dataset to a PyTorch DataLoader to load batches of data from the MNIST dataset. The images will be normalized to the range [-1, 1].
    """)
    return


@app.cell
def _(torch, torchvision):
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=data_transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    return (dataloader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The Training Function

    The training function will closely follow the pseudocode outlined above for the updated version of diffusion, where the network directly predicts the original data. The comments in the code will reference the corresponding steps in the pseudocode for clarity.
    """)
    return


@app.cell
def _(device, mo, nn, plt, torch):
    def train(net, dataloader, optimizer, num_epochs=5):
        net.train()
        loss_history = []

        for epoch in mo.status.progress_bar(range(num_epochs), title="Training Progress (Epochs)", subtitle="Warming up..."):
            for i, (x, _) in enumerate(dataloader):


                x = x.to(device) # x ~ p_{data} (The line doesn't actually sample. The sampling is done by the dataloader)
                eps = torch.randn_like(x) # eps ~ N(0, I)
                t = torch.rand(x.size(0), 1, device=device)  # t ~ Uniform(0, 1)
                z = t.view(-1, 1, 1, 1) * x + (1 - t).view(-1, 1, 1, 1) * eps # z = t * x + (1-t) * eps 
                x_pred = net(z, t) # x_pred = net(z, t)
                loss = nn.MSELoss()(x_pred, x) # loss = MSE(x_pred, x)

                # Update the neural network parameters using backpropagation and an optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if i % 50 == 0:
                    loss_history.append(loss.item())

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(loss_history, linewidth=2.5)

                    ax.set_title(f"Live Training Loss (Epoch {epoch+1}/{num_epochs})", fontsize=14, fontweight="bold")
                    ax.set_xlabel("Logging Steps (x50)", fontsize=10)
                    ax.set_ylabel("MSE Loss (x-prediction)", fontsize=10)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    mo.output.replace(
                        mo.vstack([
                            mo.md(f"**Current Step:** {i}/{len(dataloader)} | **Current Loss:** {loss.item():.4f}"),
                            mo.as_html(fig)
                        ], align="center")
                    )

                    plt.close(fig)

        mo.output.replace(
            mo.vstack([
                mo.md("### Training Complete!"),
                mo.as_html(fig)
            ], align="center")
        )

        return loss_history

    return (train,)


@app.cell
def _(UNet, dataloader, device, torch, train):
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_history = train(model, dataloader, optimizer, num_epochs=5)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The Sampling Function

    Since the model is trained, we can now sample new images from the learned distribution. Once again, we will closely follow the pseudocode outlined above.
    """)
    return


@app.cell
def _(device, torch):
    def sample(net, num_samples=8, T=50):
        net.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, 1, 28, 28, device=device) # z_0 ~ N(0, I)
            eps = z.clone()  # eps = z_0
            t = torch.linspace(0.0, 1.0, T, device=device).view(-1, 1) # t = linspace(0, 1, T)
            for i in range(T-1): # t[T] is undefined. If we used range(T), t[i+1] would be out of bounds on the last iteration
                x_pred = net(z, t[i]) # x_pred =  net(z_i, t[i])
                z = t[i+1] * x_pred + (1 - t[i+1]) * eps #  z_{i+1} = t[i+1] * x_pred + (1-t[i+1]) * eps

            z = net(z, t[T-1]).cpu()
            return z # return z_T

    return (sample,)


@app.cell(hide_code=True)
def _(mo):
    samples_slider = mo.ui.slider(start=1, stop=16, step=1, value=8, label="**Number of Samples**", show_value=True)
    steps_slider = mo.ui.slider(start=5, stop=100, step=5, value=50, label="**Sampling Steps**", show_value=True)

    refresh_btn = mo.ui.button(
        label="Regenerate", 
        value=0,
        on_click=lambda click_count: click_count + 1
    )

    controls_layout = mo.vstack([
        mo.md("### Model Inference"),
        mo.md("Adjust the parameters below to sample fresh digits from the trained model."),
        samples_slider,
        steps_slider,
        refresh_btn
    ], align="center")

    controls_layout
    return refresh_btn, samples_slider, steps_slider


@app.cell(hide_code=True)
def _(
    Image,
    mo,
    model,
    refresh_btn,
    sample,
    samples_slider,
    steps_slider,
    torch,
    torchvision,
):
    if refresh_btn.value > -1:
        with mo.status.spinner(title="Sampling..."):
                sampled_tensors = sample(model, num_samples=samples_slider.value, T=steps_slider.value)
                sampled_tensors = (sampled_tensors+1)/2
                grid_tensor = torchvision.utils.make_grid(sampled_tensors, nrow=4, normalize=False, pad_value=1)
                grid_numpy = grid_tensor.mul(255).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                final_image = Image.fromarray(grid_numpy)

                display_area = mo.vstack([
                    mo.image(final_image, width=400)
                ], align="center")

    display_area
    return


@app.cell(hide_code=True)
def _(citations_dict, mo):
    mo.md(f"""
    # Conclusion

    {citations_dict["li_back_2026"]["narrative"]} showed us that it is beneficial for diffusion models to predict the data directly rather than predicting the noise. In this blog, we first went over how diffusion models work and then showed what changes needed to be incorporated when the neural network predicts the data rather than noise. Finally, we showed how they can be implemented using Python and PyTorch.

    Hopefully, by the end of reading this, you have a better understanding of diffusion models and have the ability to implement them on your own.
    """)
    return


@app.cell(hide_code=True)
def _(bib_file, bib_keys, generate_markdown_bibliography, mo):
    mo.md(f"""
    # References

    {generate_markdown_bibliography(bib_keys, bib_file)}
    """)
    return


@app.cell(hide_code=True)
def _(citations_dict, mo):
    mo.md(f"""
    # Footnotes

    1. The notation used here is uncommon. In many papers, such as {citations_dict["ho_denoising_2020"]["narrative"]}, timestep 0 refers to the clean data point, and the data becomes *noiser* as we increase $t$. In this blog, I preferred to use opposing definitions, with $t=0$ denoting noise, so that my notation follows the paper I am focusing on and readers could move between the paper and this blog without having to change their notions of what the variables mean.
    2. The training and sampling algorithms used here actually aren't the ones used in the paper. Using the same algorithm as them would require us to first understand $v$-prediction and flow models, and I preferred not to use them for simplicity.
    """)
    return


if __name__ == "__main__":
    app.run()
