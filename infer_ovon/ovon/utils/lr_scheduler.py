from habitat import logger


class PIRLNavLRScheduler:
    def __init__(
        self,
        optimizer,
        agent,
        num_updates,
        base_lr,
        finetuning_lr,
        ppo_eps,
        start_actor_update_at,
        start_actor_warmup_at,
        start_critic_update_at,
        start_critic_warmup_at,
    ) -> None:
        self.optimizer = optimizer
        self.agent = agent
        self.update = 0
        self.num_updates = num_updates

        self.start_actor_update_at = start_actor_update_at
        self.start_actor_warmup_at = start_actor_warmup_at
        self.start_critic_update_at = start_critic_update_at
        self.start_critic_warmup_at = start_critic_warmup_at

        self.ppo_eps = ppo_eps
        self.base_lrs = [base_lr, 0, 0]
        self.finetuning_lr = finetuning_lr

        self.lr_lambdas = [
            lambda x: self.critic_linear_decay(
                x,
                start_critic_warmup_at,
                start_critic_update_at,
                base_lr,
                finetuning_lr,
            ),
            lambda x: self.linear_warmup(
                x,
                self.start_actor_warmup_at,
                self.start_actor_update_at,
                0.0,
                finetuning_lr,
            ),
            lambda x: self.linear_warmup(
                x,
                self.start_actor_warmup_at,
                self.start_actor_update_at,
                0.0,
                finetuning_lr,
            ),
        ]

    def step(self):
        self.update += 1

        if self.update == self.start_actor_warmup_at:
            logger.info(
                "\n\nAgent number of parameters pre unfreeze: {}".format(
                    sum(
                        param.numel() if param.requires_grad else 0
                        for param in self.agent.parameters()
                    )
                )
            )
            self.agent.actor_critic.unfreeze_actor()
            self.agent.actor_critic.unfreeze_state_encoder()
            logger.info(
                "\n\nAgent number of parameters post unfreeze: {}".format(
                    sum(
                        param.numel() if param.requires_grad else 0
                        for param in self.agent.parameters()
                    )
                )
            )

            logger.info("\n\nStart actor finetuning at: {}".format(self.update))

            start_index = 1
            for i, param_group in enumerate(
                self.agent.optimizer.param_groups[start_index:]
            ):
                param_group["eps"] = self.ppo_eps
                self.base_lrs[i + start_index] = 1.0

            logger.info("Base LRs: {}".format(self.base_lrs))

        if self.update == self.start_critic_warmup_at:
            self.base_lrs[0] = 1.0
            logger.info("\n\nSet critic LR at: {}".format(self.update))

        lrs = [
            base_lr * lr_lamda(self.update)
            for base_lr, lr_lamda in zip(self.base_lrs, self.lr_lambdas)
        ]

        # Set LR for each param group
        for i, data in enumerate(zip(self.optimizer.param_groups, lrs)):
            param_group, lr = data
            param_group["lr"] = lr

    def linear_warmup(
        self,
        update,
        start_update: int,
        max_updates: int,
        start_lr: int,
        end_lr: int,
    ) -> float:
        r"""
        Returns a multiplicative factor for linear value warmup
        """
        if update < start_update:
            return 1.0

        if update >= max_updates:
            return end_lr

        if max_updates == start_update:
            return end_lr

        pct_step = (update - start_update) / (max_updates - start_update)
        step_lr = (end_lr - start_lr) * pct_step + start_lr
        if step_lr > end_lr:
            step_lr = end_lr
        return step_lr

    def critic_linear_decay(
        self,
        update,
        start_update: int,
        max_updates: int,
        start_lr: int,
        end_lr: int,
    ) -> float:
        r"""
        Returns a multiplicative factor for linear value decay
        """
        if update < start_update:
            return 1

        if update >= max_updates:
            return end_lr

        if max_updates == start_update:
            return end_lr

        pct_step = (update - start_update) / (max_updates - start_update)
        step_lr = start_lr - (start_lr - end_lr) * pct_step
        if step_lr < end_lr:
            step_lr = end_lr
        return step_lr

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

        if self.update >= self.start_actor_update_at:
            self.agent.actor_critic.unfreeze_actor()
            self.agent.actor_critic.unfreeze_state_encoder()

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["optimizer", "agent", "lr_lambdas"]
        }
