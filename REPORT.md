# Assignment 2 Report

- **Name**: Arturo Velilla Martínez
- **Student ID**: 19455
- **Email**: avelilla.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem

In a large hypermarket, having an empty shelf can cost a lot of money in lost sales. But there may be a solution for this. Fixed cameras photograph every shelf section twice a day. For each photo the model decides: **pass**, **review** which means that the section manager checks the photo, or **restock**, which means a worker is sent now instead of waiting for the next scheduled restock. A photo is positive when it shows **at least k = 4 empty facings**.

I decided to do it per photo, not per gap, because stores restock by section, not hole by hole.

| Outcome                           | Cost                                | Why                                                |
|-----------------------------------|-------------------------------------|----------------------------------------------------|
| False positive (trip not needed)  | 3.25€                               | 15 minutes of a worker at 13€/h                    |
| True positive (trip needed)       | 3.25€                               | Same trip, but the sales are saved                 |
| False negative (shelf left empty) | **7.10€**                           | Approximate margin lost by having the shelf empty. |
| True negative                     | 0€                                  |                                                    |
| Review                            | 0.08€, plus 3.25€ if a trip follows | 20 seconds of a manager at 15€/h                   |

A missed restock costs **2.2 times** a wasted trip, so when there is doubt the system should tend to flag.

## Dataset

Roboflow Universe, workspace **parth-vfviv**, project **empty-shelf-detection-nipzt**, version 3 found at <https://universe.roboflow.com/parth-vfviv/empty-shelf-detection-nipzt/dataset/3>

The dataset consists of supermarket shelf photos taken from the front, labelled missing in the gaps and product on the rest. It serves the problem because the decision is a count of empty gaps, and this dataset labels exactly that, next to the products around each gap. Most alternatives only have 1 class or few images. I chose v3 because v1 and v2 have no test split, and v3 comes with Roboflow's own YOLOv8n results on its test so I could use it as reference.

## What the dataset's composition implies

| Split | Files | Unique images | missing boxes | product boxes | Images with ≥ 1 gap |
|-------|-------|---------------|---------------|---------------|---------------------|
| Train | 1,199 | 1,199         | 6,286         | 176,525       | 95%                 |
| Val   | 257   | **146**       | 954           | 22,152        | 95%                 |
| Test  | 403   | **146**       | 750           | 20,484        | 93%                 |

**Notes on the dataset:**

- **I found out that the test set is only 146 photos and not 403.** Most test and validation images are copies of others inside the same split, there are no copies in between different splits. I decided to evaluate on the 146 unique test photos and choose thresholds on the 146 unique validation photos. I think that 750 test gaps is enough.
- **Tiny, dense objects** are hard for my model. A gap only covers about 0.34% of the image (around 37px) and a product about 0.25%. Images are shrunk to 640px so upscaling doesn’t really make sense. What I did is I raised the cap from 300 to 1,000 detections per photo, because busy photos have more than 400 boxes.
- **93% of test photos have at least one gap,** so using “any image with a gap should be restocked” is useless as a strategy. That’s why I went for "at least k gaps".
- **The photos are not realistic:** They are handheld and stretched while fixed cameras would most likely be on the ceiling.

## Finetuning and baselines

I went with detection using `yolo11s`, finetuned from its COCO checkpoint. Detection because the decision needs gaps located and counted. I trained it on Google Colab for 60 epochs with patience 15 (it never triggered), image size 640, batch 16, seed 0. All models are scored on the same 146 test photos:

| Model                                                       | AP50 missing | AP50 product | mAP50     | mAP50-95  |
|-------------------------------------------------------------|--------------|--------------|-----------|-----------|
| **Finetuned yolo11s**                                       | **0.537**    | **0.913**    | **0.725** | **0.396** |
| COCO yolo11s zero-shot (every detection counted as product) | 0.000        | 0.319        | 0.159     | 0.065     |
| Predict nothing                                             | 0.000        | 0.000        | 0.000     | 0.000     |
| Reference: Roboflow's YOLOv8n                               | *0.531*      | *0.905*      | *0.718*   | *n/a*     |

The model clearly beats both baselines. COCO cannot find a single gap, because "an empty space where a product should be" is not an object COCO knows. My model is in line with Roboflow's reference and actually slightly above it. One evaluator scores all three models, using the same AP calculation as Ultralytics.

## Per-class evaluation and failure modes

| Class              | AP50  | AP50-95 | Precision | Recall |
|--------------------|-------|---------|-----------|--------|
| missing (positive) | 0.537 | 0.234   | 0.666     | 0.503  |
| product            | 0.913 | 0.559   | 0.888     | 0.854  |

The headline mAP50 of 0.725 is carried by product (0.913). Missing, which is the class that actually matters for the business, is the weak one: it only finds about half of the gaps.

| True ↓ / Predicted → | missing            | product              | not detected  |
|----------------------|--------------------|----------------------|---------------|
| missing              | **409** (55%)      | 18 (2%)              | **323 (43%)** |
| product              | 17                 | 18,712 (91%)         | 1,755 (9%)    |
| background           | **207** false gaps | 3,965 false products | n/a           |

- It almost never confuses a gap with a product (2%). **The failure is gaps that are not getting recognized.** 43% of gaps are not detected, because small dark holes look like shadow or part of the shelf. The false gaps are the same issue.
- The training curves were still slightly rising at epoch 60, so more epochs would maybe help.

## Cost matrix and operating point

| Figure | Value | Basis |
|---|---|---|
| Trip | 3.25€ | 15 min × 13€/h |
| Loss per empty facing | 0.90€ | Doing research (Gruen et al. 2002) I found out that when buyers find an empty shelf 31% buy elsewhere, 9% just don't buy. The 60% remaining just buy something else or come back later. That’s 40% of sales lost on that product for the time empty. For simplicity we will assume each empty facing brings an average of 1.5€/h.<br>So, 1.50 × 40% × 25%(margin) × 6 h = 0.90€ |
| k | 4 | Break-even 3.25€ / 0.90€ = 3.6, rounded up |
| Missed restock | 7.10€ | 0.90€ × 7.9 gaps, the average on test photos that need restocking |
| Volume | 1,500 photos/day | About 750 cameras and 2 photos, one per interval between rounds. |
| Review | 0.08€ | 20 s × 15€/h |
| Base rate | 20% | The test set has 57%, but that would mean about 850 emergency trips a day. |

I swept t from 0.00 to 1.00 on the validation photos and took the middle of the cheapest region: **t = 0.38**. On test, the model deciding alone costs **1,094€ per 1,000 photos**. It catches 44 of the 83 shelves that need restocking (53%) and sends only 2 wasted trips out of 63.

**The assumption that matters most and the hardest to estimate is the loss per gap.** Below about 3.90€ per missed restock, doing nothing is cheapest.

## What a person still has to check

I decided on a pass below **0.10**, a review between **0.10 and 0.38** and a restock above **0.38** as it was the cheapest pair on validation without exceeding 240 minutes a day.

On the test:

| Truth ↓ / Decision →     | Pass | Review | Restock |
|--------------------------|------|--------|---------|
| Needs restock (≥ 4 gaps) | 2    | 37     | 44      |
| Does not                 | 32   | 29     | 2       |

That sends 46% of the photos to review, about 686 a day. At 20 seconds each that’s 3.8 hours a day for the managers, around half a person. We have 4 hours available (15 section managers doing 15 minutes each), so it fits, but only just. If reviews take longer than 20 seconds the band would have to get narrower.

98% of shelves that need restocking get acted on (2 of 83 missed). 2 of 63 get an unnecessary trip.

| Policy                      | € per 1,000 photos | € per day | Review hours per day |
|-----------------------------|--------------------|-----------|----------------------|
| Do nothing                  | 1,420              | 2,130     | 0                    |
| Always send a trip          | 3,250              | 4,875     | 0                    |
| Model alone (t = 0.38)      | 1,094              | 1,642     | 0                    |
| A person checks every photo | **733**            | **1,100** | **8.3**              |
| **Model + review band**     | 789                | 1,184     | 3.8                  |

**Conclusion:** A person checking every photo is the cheapest option, 84€ a day less than my design, but it needs 8.3 review hours a day, more than twice the 4 hours the managers have. Among the more realistic options, model + review band is the cheapest, 946€ a day less than doing nothing and 458€ a day less than the model alone. The store's real choice is whether to hire about one extra reviewer or fit the work into the managers' existing time.

And all of this is assuming humans don’t make errors... I would go with the model.

## Limitations & next steps

1.  **Thresholds come from only 146 validation photos.** Tuned on test, the band would have moved from 0.10–0.38 to 0.13–0.56. The fix is to have more labelled photos or cross-validation.
2.  **For the reviews we assume that there is no human error**. People slacking off or having honest mistakes would happen in a real case.
3.  **The cost inputs are assumptions.** My design is based on assumptions, real data would definitely change the threshold.
4.  **Camera angle and image shape** differ from what would be in real life.
5.  **The cost of the cameras has not been accounted for,** so the savings against doing nothing are missing that part.

## Generative AI use disclosure

I used Claude Cowork to write the code, to find and check reference figures, to analyse the dataset, and to draft this report from decisions I made in conversation, which I then edited. Claude also pointed out that choosing thresholds on the test photos flattered the result. At my request it moved the choice to the validation photos. The decisions were mine: the problem, the dataset, deciding per photo with k = 4, the cost scenario, the base rate, the photo frequency, and leaving reviewer errors out of scope. Claude flagged my mistakes and proposed options with their trade-offs for several of these, and I chose among them. Claude was also used to format this report into proper .md format.
