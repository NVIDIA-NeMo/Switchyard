// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Macros used by profile implementations and profile config wiring.

macro_rules! profile_types {
    ($($config:ident),+ $(,)?) => {
        /// Resolved, strongly typed config for one profile entry.
        ///
        /// Variants intentionally mirror config type names so this macro has one
        /// input per profile and does not need a second variant-name registry.
        #[allow(clippy::enum_variant_names)]
        #[derive(Clone, Debug, PartialEq)]
        pub(crate) enum ProfileConfigEntry {
            $(
                /// Config variant generated from the profile registry.
                $config(Box<$config>),
            )+
        }

        impl ProfileConfigEntry {
            /// Returns the file-facing type discriminator for this resolved profile config.
            pub(crate) fn profile_type(&self) -> &'static str {
                match self {
                    $(
                        Self::$config(_) =>
                            <$config as crate::config::ProfileConfigDefinition>::PROFILE_TYPE,
                    )+
                }
            }

            /// Builds this resolved config into the erased runtime profile.
            pub(crate) fn build_boxed(
                &self,
            ) -> switchyard_core::Result<Box<dyn crate::Profile>> {
                match self {
                    $(
                        Self::$config(config) =>
                            <$config as crate::ProfileConfig>::build_boxed(config.as_ref()),
                    )+
                }
            }
        }

        /// Parses a serialized profile body by dispatching to the owning config type.
        ///
        /// The `type` discriminator is matched with `-` and `_` treated as
        /// equivalent, so a snake_case name copied from a legacy route bundle
        /// (e.g. `random_routing`) resolves to its v2 profile, and a hyphenated
        /// spelling of an underscore type (e.g. `stage-router`) resolves too.
        pub(crate) fn parse_profile_config(
            profile_type: &str,
            value: serde_json::Value,
            env: &crate::config::ProfileBuildEnv<'_>,
        ) -> switchyard_core::Result<ProfileConfigEntry> {
            // Compare treating `-` and `_` as equal without allocating. Profile
            // type names are short ASCII, so an equal-length per-byte scan that
            // folds the separators is enough. Registered names must stay unique
            // under this folding (they are today) so dispatch is unambiguous.
            fn eq_ignoring_separators(a: &str, b: &str) -> bool {
                a.len() == b.len()
                    && a.bytes().zip(b.bytes()).all(|(x, y)| {
                        x == y || (matches!(x, b'-' | b'_') && matches!(y, b'-' | b'_'))
                    })
            }

            $(
                if eq_ignoring_separators(
                    profile_type,
                    <$config as crate::config::ProfileConfigDefinition>::PROFILE_TYPE,
                ) {
                    let config =
                        <$config as crate::config::ProfileConfigDefinition>::parse_profile_config(
                            value,
                            env,
                        )?;
                    return Ok(ProfileConfigEntry::$config(Box::new(config)));
                }
            )+
            Err(switchyard_core::SwitchyardError::InvalidConfig(format!(
                "unknown profile type `{profile_type}`"
            )))
        }
    };
}

pub(crate) use profile_types;
