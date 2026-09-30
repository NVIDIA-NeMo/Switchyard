// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounds the actual Serde traversal without inspecting Python objects twice.

use std::fmt;

use serde::{Deserialize, Deserializer, de};

const MAX_DEPTH: usize = 128;
const DEPTH_ERROR: &str = "Python value exceeds supported nesting depth (128)";

pub(crate) fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    T::deserialize(Limited {
        inner: deserializer,
        remaining: MAX_DEPTH,
    })
}

// The budget belongs to each branch. Wide containers do not consume the depth
// available to their siblings, and ignored values are never traversed.
struct Limited<T> {
    inner: T,
    remaining: usize,
}

macro_rules! deserialize_methods {
    ($($method:ident $(($($arg:ident: $ty:ty),*))?),* $(,)?) => {
        $(
            fn $method<V>(self, $($($arg: $ty,)*)? visitor: V) -> Result<V::Value, D::Error>
            where
                V: de::Visitor<'de>,
            {
                let remaining = self.remaining.checked_sub(1).ok_or_else(|| {
                    <D::Error as de::Error>::custom(DEPTH_ERROR)
                })?;
                self.inner.$method($($($arg,)*)? Limited { inner: visitor, remaining })
            }
        )*
    };
}

impl<'de, D: Deserializer<'de>> Deserializer<'de> for Limited<D> {
    type Error = D::Error;

    deserialize_methods! {
        deserialize_any, deserialize_bool,
        deserialize_i8, deserialize_i16, deserialize_i32, deserialize_i64, deserialize_i128,
        deserialize_u8, deserialize_u16, deserialize_u32, deserialize_u64, deserialize_u128,
        deserialize_f32, deserialize_f64, deserialize_char, deserialize_str, deserialize_string,
        deserialize_bytes, deserialize_byte_buf, deserialize_option, deserialize_unit,
        deserialize_unit_struct(name: &'static str),
        deserialize_newtype_struct(name: &'static str),
        deserialize_seq, deserialize_tuple(len: usize),
        deserialize_tuple_struct(name: &'static str, len: usize), deserialize_map,
        deserialize_struct(name: &'static str, fields: &'static [&'static str]),
        deserialize_enum(name: &'static str, variants: &'static [&'static str]),
        deserialize_identifier, deserialize_ignored_any,
    }

    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }
}

macro_rules! scalar_visits {
    ($($method:ident $(($value:ident: $ty:ty))?),* $(,)?) => {
        $(
            fn $method<E>(self $(, $value: $ty)?) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.inner.$method($($value)?)
            }
        )*
    };
}

impl<'de, V: de::Visitor<'de>> de::Visitor<'de> for Limited<V> {
    type Value = V::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.expecting(formatter)
    }

    scalar_visits! {
        visit_bool(value: bool),
        visit_i8(value: i8), visit_i16(value: i16), visit_i32(value: i32),
        visit_i64(value: i64), visit_i128(value: i128),
        visit_u8(value: u8), visit_u16(value: u16), visit_u32(value: u32),
        visit_u64(value: u64), visit_u128(value: u128),
        visit_f32(value: f32), visit_f64(value: f64), visit_char(value: char),
        visit_str(value: &str), visit_borrowed_str(value: &'de str), visit_string(value: String),
        visit_bytes(value: &[u8]), visit_borrowed_bytes(value: &'de [u8]), visit_byte_buf(value: Vec<u8>),
        visit_none, visit_unit,
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        self.inner.visit_some(Limited {
            inner: deserializer,
            remaining: self.remaining,
        })
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        self.inner.visit_newtype_struct(Limited {
            inner: deserializer,
            remaining: self.remaining,
        })
    }

    fn visit_seq<A: de::SeqAccess<'de>>(self, access: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_seq(Limited {
            inner: access,
            remaining: self.remaining,
        })
    }

    fn visit_map<A: de::MapAccess<'de>>(self, access: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_map(Limited {
            inner: access,
            remaining: self.remaining,
        })
    }

    fn visit_enum<A: de::EnumAccess<'de>>(self, access: A) -> Result<Self::Value, A::Error> {
        self.inner.visit_enum(Limited {
            inner: access,
            remaining: self.remaining,
        })
    }
}

impl<'de, S: de::DeserializeSeed<'de>> de::DeserializeSeed<'de> for Limited<S> {
    type Value = S::Value;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        self.inner.deserialize(Limited {
            inner: deserializer,
            remaining: self.remaining,
        })
    }
}

impl<'de, A: de::SeqAccess<'de>> de::SeqAccess<'de> for Limited<A> {
    type Error = A::Error;

    fn next_element_seed<S: de::DeserializeSeed<'de>>(
        &mut self,
        seed: S,
    ) -> Result<Option<S::Value>, Self::Error> {
        self.inner.next_element_seed(Limited {
            inner: seed,
            remaining: self.remaining,
        })
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}

impl<'de, A: de::MapAccess<'de>> de::MapAccess<'de> for Limited<A> {
    type Error = A::Error;

    fn next_key_seed<S: de::DeserializeSeed<'de>>(
        &mut self,
        seed: S,
    ) -> Result<Option<S::Value>, Self::Error> {
        self.inner.next_key_seed(Limited {
            inner: seed,
            remaining: self.remaining,
        })
    }

    fn next_value_seed<S: de::DeserializeSeed<'de>>(
        &mut self,
        seed: S,
    ) -> Result<S::Value, Self::Error> {
        self.inner.next_value_seed(Limited {
            inner: seed,
            remaining: self.remaining,
        })
    }

    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }
}

impl<'de, A: de::EnumAccess<'de>> de::EnumAccess<'de> for Limited<A> {
    type Error = A::Error;
    type Variant = Limited<A::Variant>;

    fn variant_seed<S: de::DeserializeSeed<'de>>(
        self,
        seed: S,
    ) -> Result<(S::Value, Self::Variant), Self::Error> {
        let (value, variant) = self.inner.variant_seed(Limited {
            inner: seed,
            remaining: self.remaining,
        })?;
        Ok((
            value,
            Limited {
                inner: variant,
                remaining: self.remaining,
            },
        ))
    }
}

impl<'de, A: de::VariantAccess<'de>> de::VariantAccess<'de> for Limited<A> {
    type Error = A::Error;

    fn unit_variant(self) -> Result<(), Self::Error> {
        self.inner.unit_variant()
    }

    fn newtype_variant_seed<S: de::DeserializeSeed<'de>>(
        self,
        seed: S,
    ) -> Result<S::Value, Self::Error> {
        self.inner.newtype_variant_seed(Limited {
            inner: seed,
            remaining: self.remaining,
        })
    }

    fn tuple_variant<V: de::Visitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.tuple_variant(
            len,
            Limited {
                inner: visitor,
                remaining: self.remaining,
            },
        )
    }

    fn struct_variant<V: de::Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.inner.struct_variant(
            fields,
            Limited {
                inner: visitor,
                remaining: self.remaining,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{DEPTH_ERROR, MAX_DEPTH, deserialize};
    use serde::Deserialize;
    use serde_json::{Value, json};

    fn nested_arrays(depth: usize) -> Value {
        (0..depth).fold(Value::Null, |value, _| Value::Array(vec![value]))
    }

    #[test]
    fn depth_is_limited_per_branch() {
        let input = Value::Array(vec![
            nested_arrays(MAX_DEPTH - 2),
            nested_arrays(MAX_DEPTH - 2),
        ]);
        assert_eq!(deserialize::<Value, _>(input.clone()).unwrap(), input);

        let error = deserialize::<Value, _>(nested_arrays(MAX_DEPTH)).unwrap_err();
        assert!(error.to_string().contains(DEPTH_ERROR));
    }

    #[test]
    fn preserves_compound_enum_dispatch() {
        #[derive(Debug, Deserialize, PartialEq)]
        struct Newtype(Option<Vec<u8>>);

        #[derive(Debug, Deserialize, PartialEq)]
        enum Example {
            Unit,
            Newtype(Newtype),
            Tuple(u8, bool),
            Struct { value: Newtype },
        }

        let cases = [
            (json!("Unit"), Example::Unit),
            (
                json!({"Newtype": [1, 2]}),
                Example::Newtype(Newtype(Some(vec![1, 2]))),
            ),
            (json!({"Tuple": [3, true]}), Example::Tuple(3, true)),
            (
                json!({"Struct": {"value": null}}),
                Example::Struct {
                    value: Newtype(None),
                },
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(deserialize::<Example, _>(input).unwrap(), expected);
        }
    }

    #[test]
    fn preserves_borrowed_strings() {
        let mut input = serde_json::Deserializer::from_str("\"borrowed\"");
        assert_eq!(deserialize::<&str, _>(&mut input).unwrap(), "borrowed");
    }
}
