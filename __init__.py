# -*- coding: utf-8 -*-
"""
/***************************************************************************
 Advanced Raster Processor
                                 A QGIS Plugin
 Optimized for large point-cloud interpolation, resampling and smoothing
                              -------------------
        begin                : 2025-01-01
        copyright            :
        email                :benedikt.haimerl@uni-hamburg.de
 ***************************************************************************/

***************************************************************************
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 ***************************************************************************
"""

def classFactory(iface):
    """QGIS calls this function to load the plugin."""
    from .main import RasterProcessorPlugin
    return RasterProcessorPlugin(iface)
