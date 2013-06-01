#!/bin/bash

export COMPUTE_PROFILE=1
export COMPUTE_PROFILE_CONFIG=profile_config
export COMPUTE_PROFILE_LOG=out.profile

$1 $2 $3
