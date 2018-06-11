# GenAttackMCS2018
This is a modified implementation of generative adversarial attacks on black-box models from
GenAttack: Practical Black-box Attacks with Gradient-Free Optimization [paper](https://arxiv.org/abs/1805.11090). I used it to attack models from this [competition](https://competitions.codalab.org/competitions/19090) held as a part of [Machines Can See 2018 summit](http://machinescansee.com/).

Some notes:

* The black-box model in the competition outputs descriptors, so the original objective has been changed to MSE loss between descriptors. There are some other smaller changes made along the way.

* Apparently, the black-box model was implemented using Pytorch 0.3.1 and, unfortunately, does not go well when trying to run with Pytorch 0.4.0+ imported at the same time. So, the requirement is to use Pytorch 0.3.1.

* As the [official repo](https://github.com/nesl/adversarial_genattack) for the GenAttack paper is empty for now, in the nearest future I also plan to publish a Pytorch 0.4.0+ implementation of general purpose GenAttack, digesting both descriptors and logits, i.e. enabling the paper experiments reproduction.
