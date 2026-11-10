# Simple Guide: What Does This Project Do? (Explain Like I'm 10)

Imagine you work at an airport security gate. 
Every second, **10,000 suitcases (network packets)** come flying by on a fast conveyor belt. 
The police give you a list of **wanted items / weapons (threat signatures)** to watch out for.

Normally, checking every pocket of all 10,000 bags one-by-one by hand would take forever, and the conveyor belt would get stuck.

This project builds a **super-fast robotic scanner** using a **GPU (Graphics Card)** and smart math to check all 10,000 bags in the blink of an eye.

---

## 1. The Big Idea (The "Metal Detector" Trick)

Instead of opening every single bag and searching through every piece of cloth, we use a **two-step trick**:

```
[10,000 Bags on Conveyor Belt]
            │
            ▼
┌──────────────────────────────────────┐
│ STEP 1: The Fast Metal Detector     │
│ (GPU does quick math)                │
└──────────────────────────────────────┘
            │
            ├─► 99% of bags have ZERO metal ──► LET THEM PASS IMMEDIATELY! (Safe)
            │
            ▼
Only ~750 suspicious bags beep!
            │
            ▼
┌──────────────────────────────────────┐
│ STEP 2: The Detailed Inspection      │
│ (Super-fast CPU laser scanner)       │
└──────────────────────────────────────┘
            │
            ▼
Find the exact ~350 real bad bags and sound the alarm!
```

---

## 2. Step-by-Step: How It Works

### Step 1: Breaking Words into 3-Letter Puzzle Pieces (n-grams)
If the hacker's attack word is **`ATTACK`**, we break it into overlapping 3-letter puzzle pieces:
* `ATT`
* `TTA`
* `TAC`
* `ACK`

Each 3-letter piece gets a numbered sticker between `0` and `65,535` (this is called **Hashing**).

### Step 2: The GPU Matrix Multiplication (The Math Filter)
* **Matrix P (Packets):** A giant attendance sheet showing which puzzle pieces are in each bag.
* **Matrix S (Signatures):** A sheet showing which puzzle pieces belong to each attack.
* When we multiply them (**$C = P \times S^T$**), the math instantly counts:
  > *"How many puzzle pieces does Bag #42 share with Attack #3?"*

If an attack needs 4 pieces, and Bag #42 only has 1 piece, **it is impossible for the attack to be inside!** We throw it out instantly. We do this for all 10,000 bags at once on the GPU.

### Step 3: Checking the Short Words (The Bypass Rule)
What if a bad word is only 2 letters (like `NO`)? You can't make a 3-letter puzzle piece out of 2 letters!
* **Our smart fix:** Any word that is too short skips the math detector and goes straight to the final check. No bad words can ever slip through!

### Step 4: The 32-Byte Super Scanner (AVX2 SIMD)
For the few suspicious bags that beeped, the computer needs to read the exact text.
* Normal computers read text **1 letter at a time**.
* Our code uses special CPU powers (called **AVX2**) that grab **32 letters in one single breath**! It checks all 32 letters instantly.

---

## 3. What Did We Build in the Code?

| Fancy Name in Code | What it Actually Means |
| :--- | :--- |
| **`pcap_reader.cpp`** | The Mailman. Opens the network recording file (`.pcap`), unwraps the network envelopes (Ethernet, VLAN, IPv4, IPv6), and gets the raw message inside. |
| **`ngram_features.cpp`** | The Chopper. Chops messages into 3-letter puzzle pieces and gives each piece a number. |
| **`cuda_engine.cu`** | The Supercomputer Rocket. Runs thousands of tiny workers on the Graphics Card (NVIDIA GPU) to do all the chopping and math in milliseconds. |
| **`csr_matrix.cpp`** | The Space Saver. Since 99% of our attendance sheet is empty zeroes, this packs only the numbers so it takes 40× less memory. |
| **`cpu_engine.cpp`** | The Detective. Uses the 32-letter-at-a-time laser scanner (AVX2) to confirm if the suspicious bags really contain the attack. |
| **`alert_manager.cpp`** | The Alarm Bell. Saves a clean report (`.csv` and `.json`) listing which bad packets were caught. |

---

## 4. How Fast Is It? (The Scorecard)

We tested this with **10,000 real network packets** on an **NVIDIA T4 GPU**:

* **Normal Computer (Sequential):** Takes **2,033 milliseconds** (~2 seconds).
* **Our GPU Super Engine:** Takes only **237 milliseconds** (less than a quarter of a second!).
* **Speedup:** **8.56× faster!**
* **Throughput:** Scans **42,075 packets every single second!**
* **Accuracy:** Caught **355 out of 355 attacks (100% correct)**, with **ZERO missed attacks** and **ZERO false alarms**.

---

## 5. Cheat Sheet for Your Viva / Exam (Say This in Simple Words!)

1. **"What is the main goal of your project?"**
   > *"We made Deep Packet Inspection (finding viruses/attacks in network traffic) 8.5 times faster by turning text searching into sparse matrix math on a GPU."*

2. **"Why use matrix math instead of normal text search?"**
   > *"Normal text search jumps all over computer memory and makes GPUs slow. Matrix math is what GPUs are born to do—they can crunch all 10,000 packets together in parallel."*

3. **"Can your scanner miss an attack?"**
   > *"No. If an attack word is inside a packet, all its 3-letter pieces must also be inside. If the math shows pieces are missing, it's physically impossible for the attack to be there. And short words skip the filter so nothing is missed."*

4. **"What made it jump to 8.56× speedup?"**
   > *"Two things: We moved the feature chopping directly into GPU memory so we didn't waste time copying data, and we used AVX2 CPU instructions to scan 32 bytes at a time in the final verification step."*
