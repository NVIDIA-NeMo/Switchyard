// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const crypto = require('node:crypto');
const vscode = require('vscode');
const customModels = require('./custom-models.js');
const { toCopilot, toSwitchyard } = require('./messages.js');
const { selectTier } = require('./router.js');

// Profiles and custom models use application scope, so a workspace cannot
// point a saved API key at a different endpoint.
const settings = () => vscode.workspace.getConfiguration('switchyard');
const update = (key, value) => settings().update(key, value, vscode.ConfigurationTarget.Global);
const apiKeyName = id => `switchyard.apiKey.${id}`;
const defaultName = (capable, efficient) => `Switchyard: ${capable.name} / ${efficient.name}`;

class Provider {
  constructor(context, log) {
    this.context = context;
    this.log = log;
    this.changed = new vscode.EventEmitter();
    this.onDidChangeLanguageModelChatInformation = this.changed.event;
  }

  // Copilot models and custom endpoints. Both have the same sendRequest and countTokens methods.
  // Copilot counts tokens many times per request, so the list is cached until models or settings change.
  models() {
    const pending = this.cache ??= (async () => {
      const copilot = await vscode.lm.selectChatModels({ vendor: 'copilot' });
      const custom = await Promise.all(settings().get('customModels', []).map(async config =>
        customModels.createModel(config, await this.context.secrets.get(apiKeyName(config.id)))));
      // Copilot's own "auto" model is a router, not a model.
      return [...copilot.filter(model => model.id !== 'auto'), ...custom];
    })();
    // Do not keep a failed lookup, for example while Copilot is still starting.
    pending.catch(() => {
      if (this.cache === pending) this.cache = undefined;
    });
    return pending;
  }

  refresh() {
    this.cache = undefined;
    this.changed.fire();
  }

  async provideLanguageModelChatInformation() {
    const models = await this.models();
    const result = [];
    for (const profile of settings().get('profiles', [])) {
      const pair = findPair(profile, models);
      if (!pair) continue;
      result.push({
        id: profile.id,
        name: profile.name,
        family: 'switchyard',
        version: '1',
        tooltip: `Capable: ${pair.capable.name}\nEfficient: ${pair.efficient.name}`,
        maxInputTokens: Math.min(pair.capable.maxInputTokens, pair.efficient.maxInputTokens),
        maxOutputTokens: customModels.MAX_OUTPUT_TOKENS,
        capabilities: { toolCalling: true, imageInput: false },
      });
    }
    return result;
  }

  async provideTokenCount(info, text, token) {
    const pair = await this.profile(info.id);
    const input = typeof text === 'string' ? text : toCopilot([text])[0];
    if (!input) return 0;
    const counts = await Promise.all([pair.capable, pair.efficient].map(model => model.countTokens(input, token)));
    return Math.max(...counts);
  }

  async provideLanguageModelChatResponse(info, messages, options, progress, token) {
    const { profile, ...pair } = await this.profile(info.id);
    // VS Code passes the chat ID in an internal field. Without it, each request is routed on its own.
    const sessionId = options.modelOptions?._conversationId;
    const tier = await selectTier(profile.threshold ?? 0.5, toSwitchyard(messages), sessionId);
    const model = pair[tier];
    this.log.info(`${profile.name}: ${tier} -> ${model.name}`);

    const input = model.vendor === 'copilot' ? toCopilot(messages) : messages;
    const response = await model.sendRequest(input, {
      tools: options.tools,
      toolMode: options.toolMode,
      justification: `Switchyard routes ${profile.name} requests to this model.`,
    }, token);
    for await (const part of response.stream) {
      if (part instanceof vscode.LanguageModelTextPart || part instanceof vscode.LanguageModelToolCallPart) {
        progress.report(part);
      }
    }
  }

  // Returns the profile with its capable and efficient models.
  async profile(id) {
    const profile = settings().get('profiles', []).find(p => p.id === id);
    const pair = profile && findPair(profile, await this.models());
    if (!pair) throw new Error('This Switchyard profile uses a model that is not available.');
    return { profile, ...pair };
  }
}

function findPair(profile, models) {
  const capable = models.find(model => model.id === profile.capable);
  const efficient = models.find(model => model.id === profile.efficient);
  return capable && efficient ? { capable, efficient } : undefined;
}

async function pickModel(models, title, currentId) {
  const items = models.map(model => ({
    label: model.name,
    description: model.id === currentId ? 'current' : model.vendor === 'copilot' ? 'Copilot' : 'Custom',
    model,
  }));
  return (await vscode.window.showQuickPick(items, { title, ignoreFocusOut: true }))?.model;
}

// Asks for the two models and a name. Returns undefined if the user cancels.
async function profileForm(models, previous = {}) {
  const capable = await pickModel(models, 'Capable model (for hard turns)', previous.capable);
  if (!capable) return undefined;
  const others = models.filter(model => model.id !== capable.id);
  const efficient = await pickModel(others, 'Efficient model (for routine turns)', previous.efficient);
  if (!efficient) return undefined;

  const name = await vscode.window.showInputBox({
    title: 'Profile name',
    value: previous.name ?? defaultName(capable, efficient),
    prompt: 'This name appears in the chat model picker.',
    ignoreFocusOut: true,
  });
  if (name === undefined) return undefined;
  return { ...previous, name: name.trim() || defaultName(capable, efficient), capable: capable.id, efficient: efficient.id };
}

async function addCustomModel(context) {
  const format = await vscode.window.showQuickPick(customModels.FORMATS, { title: 'API format', ignoreFocusOut: true });
  if (!format) return;
  const url = await vscode.window.showInputBox({
    title: 'Endpoint URL', placeHolder: format.example, validateInput: customModels.checkUrl, ignoreFocusOut: true,
  });
  if (!url) return;
  const model = await vscode.window.showInputBox({ title: 'Model ID', prompt: 'The model name the endpoint expects.', ignoreFocusOut: true });
  if (!model) return;
  const apiKey = await vscode.window.showInputBox({
    title: 'API key', prompt: 'Leave blank if the endpoint needs no key. The key is stored in VS Code secret storage.',
    password: true, ignoreFocusOut: true,
  });
  if (apiKey === undefined) return;
  const contextSize = await vscode.window.showInputBox({
    title: 'Context size in tokens', value: '128000', prompt: 'Check the model documentation for this number.', ignoreFocusOut: true,
    validateInput: value => Number.isInteger(Number(value)) && Number(value) >= 8192 ? undefined : 'Enter a whole number of at least 8192.',
  });
  if (!contextSize) return;
  const name = await vscode.window.showInputBox({ title: 'Display name', value: model, ignoreFocusOut: true });
  if (!name) return;

  const config = { id: crypto.randomUUID().slice(0, 8), name, model, url, format: format.value, maxInputTokens: Number(contextSize) };
  const reached = await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: `Testing ${name}...`, cancellable: true },
    async (_progress, token) => {
      try {
        await customModels.checkModel(config, apiKey, token);
        return true;
      } catch (error) {
        if (!token.isCancellationRequested) vscode.window.showErrorMessage(`Could not reach ${name}: ${error.message}`);
        return false;
      }
    },
  );
  if (!reached) return;
  await context.secrets.store(apiKeyName(config.id), apiKey);
  await update('customModels', [...settings().get('customModels', []), config]);
  vscode.window.showInformationMessage(`Added ${name}. You can now use it in a routing profile.`);
}

async function removeCustomModel(context) {
  const list = settings().get('customModels', []);
  const picked = await vscode.window.showQuickPick(
    list.map(config => ({ label: config.name, description: config.url, config })),
    { title: 'Remove custom model' },
  );
  if (!picked) return;
  await update('customModels', list.filter(config => config.id !== picked.config.id));
  await context.secrets.delete(apiKeyName(picked.config.id));
}

async function createProfile(provider) {
  const models = await provider.models();
  if (models.length < 2) throw new Error('Two models are needed. Sign in to Copilot or add a custom model.');
  const profile = await profileForm(models, { id: crypto.randomUUID().slice(0, 8) });
  if (!profile) return;
  await update('profiles', [...settings().get('profiles', []), profile]);
  const choice = await vscode.window.showInformationMessage(
    `Saved "${profile.name}". Choose it in the chat model picker.`, 'Not in the picker?');
  if (choice) await offerEnterpriseWorkaround();
}

async function editProfile(provider) {
  const list = settings().get('profiles', []);
  const workaround = settings().get('enterpriseWorkaround');
  const items = [
    { label: '$(add) Create a new profile' },
    ...list.map(profile => ({ label: profile.name, profile })),
    { label: '', kind: vscode.QuickPickItemKind.Separator },
    {
      label: `${workaround ? '$(check) ' : ''}Enterprise workaround`,
      description: workaround ? 'On. Select to turn off.' : 'Off. Select to turn on.',
      toggle: true,
    },
  ];
  const picked = await vscode.window.showQuickPick(items, { title: 'Edit routing profile' });
  if (!picked) return;
  if (picked.toggle) return toggleEnterpriseWorkaround(!workaround);
  if (!picked.profile) return createProfile(provider);

  const action = await vscode.window.showQuickPick(['Change models or name', 'Delete'], { title: picked.profile.name });
  if (action === 'Delete') {
    await update('profiles', list.filter(profile => profile.id !== picked.profile.id));
  } else if (action) {
    const profile = await profileForm(await provider.models(), picked.profile);
    if (!profile) return;
    await update('profiles', list.map(p => (p.id === profile.id ? profile : p)));
  }
}

async function offerEnterpriseWorkaround() {
  const choice = await vscode.window.showInformationMessage(
    'Copilot Business and Enterprise can hide models from extensions. Ask your admin to allow them. '
    + 'You can also turn on an unsupported workaround that shows them in this VS Code. '
    + 'It overrides your organization\'s Copilot policy.',
    { modal: true }, 'Turn On Workaround');
  if (choice) await toggleEnterpriseWorkaround(true);
}

async function toggleEnterpriseWorkaround(on) {
  await update('enterpriseWorkaround', on);
  // Turning it off takes effect on reload, when Copilot sets its own value again.
  const message = on ? 'Enterprise workaround is on.' : 'Enterprise workaround is off. Reload the window to apply.';
  const choice = await vscode.window.showInformationMessage(message, ...(on ? [] : ['Reload Window']));
  if (choice) await vscode.commands.executeCommand('workbench.action.reloadWindow');
}

// Copilot Business and Enterprise hide extension models when the organization
// has not allowed them. This sets Copilot's internal flag that shows them, which
// overrides the organization's policy. This is known and opt-in.
// Copilot resets the flag, so it is set again every few seconds.
function startEnterpriseWorkaround(context) {
  const apply = () => {
    if (settings().get('enterpriseWorkaround')) {
      vscode.commands.executeCommand('setContext', 'github.copilot.clientByokEnabled', true);
    }
  };
  const timer = setInterval(apply, 5000);
  context.subscriptions.push({ dispose: () => clearInterval(timer) });
  apply();
}

function activate(context) {
  const log = vscode.window.createOutputChannel('Switchyard', { log: true });
  const provider = new Provider(context, log);
  const command = (name, handler) => vscode.commands.registerCommand(name, async () => {
    try {
      await handler();
    } catch (error) {
      vscode.window.showErrorMessage(`Switchyard: ${error.message}`);
    }
  });

  // Refresh the picker when Copilot's models change. Compare IDs so that our
  // own refresh does not cause another one.
  let copilotIds = '';
  const refreshOnCopilotChange = () => vscode.lm.selectChatModels({ vendor: 'copilot' }).then(models => {
    const ids = models.map(model => model.id).join();
    if (ids !== copilotIds) {
      copilotIds = ids;
      provider.refresh();
    }
  }, error => log.error(error));

  context.subscriptions.push(
    log,
    provider.changed,
    vscode.lm.registerLanguageModelChatProvider('switchyard', provider),
    vscode.lm.onDidChangeChatModels(refreshOnCopilotChange),
    vscode.workspace.onDidChangeConfiguration(event => {
      if (event.affectsConfiguration('switchyard')) provider.refresh();
    }),
    command('switchyard.createProfile', () => createProfile(provider)),
    command('switchyard.editProfile', () => editProfile(provider)),
    command('switchyard.viewProfiles', () =>
      vscode.commands.executeCommand('workbench.action.chat.manage', '@provider:switchyard')),
    command('switchyard.addCustomModel', () => addCustomModel(context)),
    command('switchyard.removeCustomModel', () => removeCustomModel(context)),
    command('switchyard.showLog', () => log.show()),
  );
  startEnterpriseWorkaround(context);
}

module.exports = { activate };
