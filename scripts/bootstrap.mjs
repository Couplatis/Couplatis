import which from "which";
import chalk from "chalk";
import fs from "fs";
import { execSync, spawnSync } from "child_process";

const CPU = process.argv.includes("--cpu");
const RERUN = !process.argv.includes("--no-rerun");

function info(...msgs) {
  console.log(chalk.green("[bootstrap]"), ...msgs);
}

function error(...msgs) {
  console.error(chalk.red("[error]"), ...msgs);
}

function warn(...msgs) {
  console.warn(chalk.yellow("[warning]"), ...msgs);
}

async function findConda() {
  return await which("conda", {
    nothrow: true,
  });
}

async function findCuda() {
  return await which("nvcc", {
    nothrow: true,
  });
}

async function findPython() {
  return await which("python", {
    nothrow: true,
  });
}

async function findPDM() {
  return await which("pdm", {
    nothrow: true,
  });
}

async function findPip() {
  return await which("pip", {
    nothrow: true,
  });
}

async function findPipx() {
  return await which("pipx", {
    nothrow: true,
  });
}

async function findShell() {
  if (process.env.SHELL) {
    return process.env.SHELL;
  }
  const shells = ["pwsh", "powershell", "cmd", "bash", "zsh"];

  for (const shell of shells) {
    if (await which(shell, { nothrow: true })) {
      return shell;
    }
  }
}

async function checkPDM() {
  const pdm = await findPDM();
  if (pdm) {
    info("PDM found at", chalk.yellow(pdm));
    return true;
  }
  const pip = await findPip();
  if (!(await findPipx()) && pip) {
    info("Installing pipx...");
    execSync("pip install pipx", { stdio: "inherit" });
  } else {
    if (!pip) {
      error("Python pip not found, please install pip and try again.");
      process.exit(1);
    }
  }
  info("Installing PDM...");
  execSync("pipx ensurepath", { stdio: "inherit" });
  execSync("pipx install pdm", { stdio: "inherit" });
  info("PDM installed successfully.");
  return true;
}

async function checkCondaEnv() {
  const condaEnv = JSON.parse(
    spawnSync("conda", ["env", "list", "--json", "--quiet"], {
      encoding: "utf-8",
      stdio: "pipe",
    }).stdout
  );
  let env;
  for (const _env of condaEnv.envs) {
    if (_env.includes("couplatis")) {
      env = _env;
      break;
    }
  }
  if (env) {
    info(
      "Conda couplatis virtual environment found at",
      chalk.yellow(env) + ", skipping creation."
    );
    return true;
  }
}

async function validateCondaEnv() {
  const condaEnv = JSON.parse(
    spawnSync("conda", ["env", "list", "--json", "--quiet"], {
      encoding: "utf-8",
      stdio: "pipe",
    }).stdout
  );
  console.log(condaEnv);
  return condaEnv.active_prefix_name == "couplatis";
}

function getCudaVersion() {
  const nvccVersion = spawnSync("nvcc", ["--version"], {
    encoding: "utf-8",
    stdio: "pipe",
  }).stdout;
  return nvccVersion.match(/(\d+\.\d+)/)[0];
}

function validateCudaVersion(cudaVersion) {
  if (cudaVersion > "12.4") {
    return false;
  } else {
    return true;
  }
}

function checkTorchEnv() {
  const torch = spawnSync(
    "python",
    ["-c", "import torch; print(torch.__version__)"],
    {
      encoding: "utf-8",
      stdio: "pipe",
    }
  );
  if (torch.status ?? 0) {
    error(torch.stderr);
    return false;
  }
  if (!torch.stdout.includes("2.4")) {
    error(
      "Invalid torch version found, perhaps your conda installation is broken, recreate couplatis conda environment manually and try again."
    );
    process.exit(1);
  }
  return true;
}

async function main() {
  info("Bootstrapping Couplatis...");
  const conda = await findConda();
  if (!conda) {
    error(`Conda not found. Please install conda and try again.`);
    return;
  }
  info("Conda found at", chalk.yellow(conda));
  info("Checking for PDM installation...");
  await checkPDM();
  if (!CPU) {
    info("Checking CUDA installation...");
    const cuda = await findCuda();
    if (!cuda) {
      error(`CUDA not found. Please install CUDA and try again.`);
      process.exit(1);
    }
    const cudaVersion = getCudaVersion();
    info(
      "Found CUDA",
      chalk.greenBright(cudaVersion),
      "at",
      chalk.yellow(cuda)
    );
    if (!validateCudaVersion(cudaVersion)) {
      error(
        `CUDA version ${cudaVersion} is not supported. Please install CUDA 12.4 or previous and try again.`
      );
    }
  } else {
    warn("CPU mode enabled, skipping CUDA installation check.");
  }

  info("Checking conda environment...");
  if (!(await checkCondaEnv())) {
    info("Creating conda couplatis virtual environment...");
    spawnSync("conda", ["create", "-n", "couplatis", "python==3.12", "-y"], {
      encoding: "utf-8",
      stdio: "inherit",
    });
    info("Conda couplatis virtual environment created.");
  }

  info("Validating conda couplatis virtual environment...");
  if (!(await validateCondaEnv())) {
    if (!RERUN) {
      error(
        "Conda couplatis virtual environment not activated, please activate it and try again."
      );
      process.exit(1);
    }
    warn(
      "Conda couplatis virtual environment not activated, activating and retrying..."
    );
    const shell = await findShell();
    if (!shell) {
      error("Unrecognized shell.");
      process.exit(1);
    }
    const re_run = spawnSync(
      shell,
      [
        "-c",
        "conda activate couplatis && pnpm bootstrap --no-rerun" + (CPU ? " --cpu" : ""),
      ],
      { stdio: "inherit" }
    );
    process.exit(re_run.status ?? 0);
  }

  info("Checking Torch installation...");
  const python = await findPython();
  if (!python || !python.includes("couplatis")) {
    error(
      `Couplatis virtual python not found. Perhaps your conda installation is broken, recreate couplatis conda environment manually and try again.`
    );
    process.exit(1);
  }
  if (!checkTorchEnv()) {
    info("Installing Torch environment...");
    if (CPU) {
      warn(
        "CPU mode enabled, will not install GPU version of Torch environment."
      );
    }
    const install = spawnSync(
      "conda",
      CPU
        ? [
            "install",
            "pytorch",
            "torchvision",
            "torchaudio",
            "cpuonly",
            "-c",
            "pytorch",
            "-y",
          ]
        : [
            "install",
            "pytorch",
            "torchvision",
            "torchaudio",
            `pytorch-cuda=${getCudaVersion()}`,
            "-c",
            "pytorch",
            "-c",
            "nvidia",
            "-y",
          ],
      {
        encoding: "utf-8",
        stdio: "inherit",
      }
    );
    if ((install.status ?? 0) !== 0) {
      error(install.stderr);
      error(
        "Installation failed. Please check your conda installation and make sure to run",
        chalk.yellowBright("`conda init`"),
        "for your shell, if you perhaps this is a mistake, please retry to run",
        chalk.yellowBright("`pnpm bootstrap`"),
        "again."
      );
      process.exit(1);
    }
    info("Validating Torch installation...");

    if (!checkTorchEnv()) {
      error("Installation failed by unexpected reason.");
      process.exit(1);
    }
    info("Torch environment installed successfully.");
  }
  info("Valid Torch installation found.");
  info("Setting up pdm...");
  fs.writeFileSync(".pdm-python", python, { encoding: "utf-8" });
  const pdm = spawnSync("pdm", ["install"], {
    encoding: "utf-8",
    stdio: "pipe",
  });
  if ((pdm.status ?? 0) !== 0) {
    error(pdm.stderr);
    error("Dependency installation failed.");
    process.exit(1);
  }
  info("Dependency installation completed.");
  info("Bootstrap completed.");
}

main();
